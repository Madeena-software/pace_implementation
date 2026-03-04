"""
Fast and Adaptive Bidimensional Empirical Mode Decomposition (FABEMD)
=====================================================================

Implementation based on:
    Bhuiyan, S.M.A., Adhami, R.R., Khan, J.F. (2008).
    "A novel approach of fast and adaptive bidimensional empirical mode
    decomposition." IEEE International Conference on Acoustics, Speech
    and Signal Processing (ICASSP 2008).

Key features:
- Adaptive window sizing based on extrema spatial distribution
- Order-statistics filters (MAX/MIN) for envelope estimation (no surface
  interpolation needed)
- Envelope smoothing via averaging filters
- GPU-accelerated using CuPy when available, falls back to NumPy/SciPy

Compatible with the `BEMD` interface in `image_pipeline.py`.
Compatible with Google Colab (GPU or CPU runtime).

Author: Refactored and improved from Bhuiyan et al. (2008) paper
"""

import gc
import logging
from typing import List, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# GPU / CPU compatibility layer
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from cupyx.scipy.ndimage import (
        maximum_filter as _maximum_filter,
        minimum_filter as _minimum_filter,
        uniform_filter as _uniform_filter,
    )
    HAS_CUPY = True
except ImportError:
    from scipy.ndimage import (
        maximum_filter as _maximum_filter,
        minimum_filter as _minimum_filter,
        uniform_filter as _uniform_filter,
    )
    HAS_CUPY = False


def _xp():
    """Return the active array module (cupy or numpy)."""
    return cp if HAS_CUPY else np


def _to_numpy(arr) -> np.ndarray:
    """Convert an array to NumPy regardless of backend."""
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return arr.get()
    return np.asarray(arr)


def _to_xp(arr):
    """Convert an array to the active backend."""
    xp = _xp()
    if xp is np:
        return np.asarray(arr)
    return cp.asarray(arr)


logger = logging.getLogger(__name__)


class FABEMD:
    """
    Fast and Adaptive Bidimensional Empirical Mode Decomposition.

    This class implements the FABEMD algorithm which decomposes a 2-D signal
    (image) into a finite set of Bidimensional Intrinsic Mode Functions (BIMFs)
    and a residue.

    The key innovation from Bhuiyan et al. (2008) is the use of
    **order-statistics filters** with **adaptive window sizes** determined by
    the spatial distribution of local extrema, replacing expensive surface
    interpolation methods used in traditional BEMD.

    Algorithm outline (per BIMF):
        1. Find local maxima / minima via morphological comparison.
        2. Compute adaptive window sizes from nearest-neighbour distances
           among the extrema points.
        3. Estimate upper envelope (MAX filter) and lower envelope (MIN filter)
           with the computed window sizes.
        4. Smooth envelopes using an averaging (uniform) filter.
        5. Mean envelope = (upper + lower) / 2.
        6. Subtract mean envelope → candidate BIMF.
        7. Repeat sifting until the standard-deviation criterion is satisfied.
        8. Extract BIMF, update residual, continue until the residual has
           too few extrema.

    Parameters
    ----------
    max_sift_iterations : int
        Maximum number of sifting iterations per BIMF extraction.
    sd_threshold : float
        Stopping criterion for sifting: when SD < sd_threshold the current
        iterate is accepted as a BIMF.
    min_extrema : int
        Stop decomposition when the residual has fewer than this many
        extrema points.
    max_bimfs : int
        Hard limit on the number of BIMFs to extract.
    initial_window_size : int or None
        If given, use a fixed window size instead of adaptive computation
        (useful for reproducibility / speed).
    window_size_cap : int
        Upper bound for the adaptive window size to prevent excessively
        large filters on sparse extrema distributions.
    extrema_window : int
        Neighbourhood size used when detecting local maxima/minima.
    """

    def __init__(
        self,
        max_sift_iterations: int = 10,
        sd_threshold: float = 0.2,
        min_extrema: int = 5,
        max_bimfs: int = 100,
        initial_window_size: Optional[int] = None,
        window_size_cap: int = 201,
        extrema_window: int = 3,
    ):
        self.max_sift_iterations = max_sift_iterations
        self.sd_threshold = sd_threshold
        self.min_extrema = min_extrema
        self.max_bimfs = max_bimfs
        self.initial_window_size = initial_window_size
        self.window_size_cap = window_size_cap
        self.extrema_window = extrema_window

    # ------------------------------------------------------------------
    # Extrema detection
    # ------------------------------------------------------------------
    @staticmethod
    def _find_local_extrema(
        image, window_size: int = 3
    ):
        """
        Detect local maxima and minima maps using morphological comparison.

        A pixel is a local maximum if it equals the maximum in its
        neighbourhood **and** is non-zero.  Analogously for minima.

        Parameters
        ----------
        image : ndarray (cupy or numpy)
            Input 2-D array.
        window_size : int
            Neighbourhood size for the comparison.

        Returns
        -------
        max_map, min_map : ndarray (bool)
            Boolean masks of detected maxima / minima.
        """
        mask = image != 0
        max_map = (image == _maximum_filter(image, size=window_size)) & mask
        min_map = (image == _minimum_filter(image, size=window_size)) & mask
        return max_map, min_map

    # ------------------------------------------------------------------
    # Adaptive window size computation (Bhuiyan et al. §III-B)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_adaptive_window_size(
        extrema_map,
        image_shape: Tuple[int, int],
        cap: int = 201,
        max_sample: int = 2000,
    ) -> int:
        """
        Compute an adaptive window size from the extrema spatial distribution.

        The window is based on the *maximum nearest-neighbour distance* among
        the detected extrema, ensuring it is large enough to bridge the widest
        gap between adjacent extrema and thus produce a smooth envelope.

        For efficiency, if there are more than ``max_sample`` extrema the
        computation is done on a random sub-sample.

        Parameters
        ----------
        extrema_map : ndarray (bool)
            Boolean mask of extrema positions.
        image_shape : tuple of int
            (rows, cols) of the image.
        cap : int
            Upper bound on the returned window size.
        max_sample : int
            Maximum number of extrema to use for distance computation.

        Returns
        -------
        w : int
            Odd-valued adaptive window size (clamped to [3, cap]).
        """
        xp = _xp()
        # Get extrema coordinates on CPU for spatial analysis
        positions = _to_numpy(xp.argwhere(extrema_map))  # (N, 2)
        n = len(positions)

        if n < 2:
            # Not enough extrema → fall back to image-fraction heuristic
            return min(max(3, int(min(image_shape) * 0.1) | 1), cap)

        # Sub-sample if too many extrema (for speed)
        if n > max_sample:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_sample, replace=False)
            positions = positions[idx]
            n = max_sample

        # --- Nearest-neighbour distances via chunked computation ----------
        # Avoid (N x N) memory blow-up by processing in chunks.
        chunk = 512
        nn_dists = np.full(n, np.inf)
        for i in range(0, n, chunk):
            pts_i = positions[i : i + chunk]  # (C, 2)
            # Broadcast: (C, 1, 2) - (1, N, 2) → (C, N, 2)
            diff = pts_i[:, None, :] - positions[None, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))  # (C, N)
            # Exclude self-distance (set diagonal-block to inf)
            for k in range(len(pts_i)):
                dists[k, i + k] = np.inf
            nn_dists[i : i + chunk] = dists.min(axis=1)

        max_nn_dist = float(np.max(nn_dists))

        # Window = 2 * ceil(max_nn_dist) + 1  (must be odd)
        w = 2 * int(np.ceil(max_nn_dist)) + 1
        w = max(3, min(w, cap))
        # Ensure odd
        if w % 2 == 0:
            w += 1
        return w

    # ------------------------------------------------------------------
    # Envelope estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_envelope(
        image,
        window_size: int,
        filter_type: str = "max",
    ):
        """
        Estimate an envelope surface using an order-statistics filter
        followed by averaging (smoothing).

        Parameters
        ----------
        image : ndarray (cupy or numpy)
            Input 2-D signal.
        window_size : int
            Window size for both the order-statistics filter and the
            subsequent smoothing filter.
        filter_type : str
            ``"max"`` for upper envelope, ``"min"`` for lower envelope.

        Returns
        -------
        smoothed_envelope : ndarray
        """
        if filter_type == "max":
            envelope = _maximum_filter(image, size=window_size)
        elif filter_type == "min":
            envelope = _minimum_filter(image, size=window_size)
        else:
            raise ValueError("filter_type must be 'max' or 'min'")
        # Smooth the envelope with an averaging filter
        return _uniform_filter(envelope, size=window_size)

    # ------------------------------------------------------------------
    # SD stopping criterion
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_sd(prev, curr) -> float:
        """
        Compute the normalised standard-deviation between successive sifting
        iterates (Huang et al. criterion).

        SD = sum((prev - curr)^2) / sum(prev^2)
        """
        xp = _xp()
        denom = xp.sum(prev ** 2)
        if float(denom) == 0:
            return 0.0
        return float(xp.sum((curr - prev) ** 2) / denom)

    # ------------------------------------------------------------------
    # Main decomposition
    # ------------------------------------------------------------------
    def decompose(self, image) -> list:
        """
        Perform FABEMD decomposition on an input image.

        Parameters
        ----------
        image : ndarray (cupy or numpy)
            Input 2-D image (will be cast to float64 internally).

        Returns
        -------
        bimfs : list of ndarray
            Extracted BIMFs ordered from highest to lowest frequency.
        """
        xp = _xp()
        logger.info("Starting FABEMD decomposition...")

        image = _to_xp(image)
        residual = image.astype(xp.float64)
        bimfs: list = []

        while len(bimfs) < self.max_bimfs:
            h = residual.copy()
            sd_limit = self.sd_threshold
            accepted = False

            for j in range(self.max_sift_iterations):
                # 1. Detect extrema
                max_map, min_map = self._find_local_extrema(
                    h, window_size=self.extrema_window
                )

                # 2. Adaptive window sizes (per the paper)
                if self.initial_window_size is not None:
                    w_upper = self.initial_window_size
                    w_lower = self.initial_window_size
                else:
                    w_upper = self._compute_adaptive_window_size(
                        max_map, h.shape, cap=self.window_size_cap
                    )
                    w_lower = self._compute_adaptive_window_size(
                        min_map, h.shape, cap=self.window_size_cap
                    )

                # 3. Estimate envelopes
                upper_env = self._estimate_envelope(h, w_upper, "max")
                lower_env = self._estimate_envelope(h, w_lower, "min")

                # 4. Mean envelope & candidate BIMF
                mean_env = (upper_env + lower_env) / 2.0
                h_next = h - mean_env

                # 5. SD criterion
                sd = self._compute_sd(h, h_next)

                if sd < sd_limit:
                    bimfs.append(h_next)
                    residual = residual - h_next
                    accepted = True
                    break
                else:
                    sd_limit *= 1.1  # relax threshold slightly
                    h = h_next

            if not accepted:
                # Max sifting iterations reached — accept last iterate
                bimfs.append(h)
                residual = residual - h

            # 6. Check stopping criteria on residual
            max_map, min_map = self._find_local_extrema(residual)
            n_extrema = int(xp.sum(max_map) + xp.sum(min_map))
            print(
                f"\rFABEMD: {len(bimfs)} BIMFs | "
                f"residual extrema: {n_extrema}",
                end="",
            )

            if n_extrema <= self.min_extrema:
                logger.info(
                    f"Stopping: residual has ≤ {self.min_extrema} extrema "
                    f"({n_extrema})."
                )
                break

            del max_map, min_map
            gc.collect()

        print()  # newline after progress
        logger.info(
            f"FABEMD decomposition completed — {len(bimfs)} BIMFs extracted."
        )
        return bimfs

    # ------------------------------------------------------------------
    # Utility methods (compatible with BEMD interface)
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_energies(bimfs: list) -> List[float]:
        """Calculate energy (sum of squares) of each BIMF."""
        xp = _xp()
        return [float(xp.sum(b ** 2)) for b in bimfs]

    @staticmethod
    def calculate_entropy(bimf, bins: int = 256) -> float:
        """
        Calculate Shannon entropy of a single BIMF.

        The BIMF values are quantised into *bins* levels and the entropy
        is computed from the resulting histogram.

        Parameters
        ----------
        bimf : ndarray (cupy or numpy)
            A single BIMF (2-D array).
        bins : int
            Number of histogram bins.

        Returns
        -------
        entropy : float
            Shannon entropy in nats.
        """
        data = bimf.ravel()
        # Move to CPU for histogram
        data_cpu = _to_numpy(data)
        hist, _ = np.histogram(data_cpu, bins=bins)
        # Normalise to probability
        p = hist / hist.sum()
        # Remove zeros for log stability
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def calculate_all_entropies(
        bimfs: list, bins: int = 256
    ) -> List[float]:
        """Calculate Shannon entropy for every BIMF in the list."""
        return [FABEMD.calculate_entropy(b, bins=bins) for b in bimfs]

    @staticmethod
    def calculate_entropy_16bit(image: np.ndarray) -> float:
        """
        Calculate entropy using 16-bit histogram (compatible with
        ``ImageMetrics.calculate_entropy`` in image_pipeline).
        """
        hist = np.histogram(image.ravel(), bins=65535, range=(0, 65535))[0]
        p = hist / hist.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    # ------------------------------------------------------------------
    # Residual accessor
    # ------------------------------------------------------------------
    def decompose_with_residual(
        self, image
    ) -> Tuple[list, np.ndarray]:
        """
        Decompose and also return the final residue.

        Returns
        -------
        bimfs : list of ndarray
        residual : ndarray
        """
        xp = _xp()
        image = _to_xp(image)
        bimfs = self.decompose(image)
        residual = image.astype(xp.float64) - sum(bimfs)
        return bimfs, residual
