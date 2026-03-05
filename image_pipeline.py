"""
Image Processing Pipeline
==========================
A unified, class-based image processing pipeline for medical imaging.

This module consolidates all image processing functionality including:
- Flat Field Correction (FFC)
- Spatial Calibration
- BEMD (Bidimensional Empirical Mode Decomposition)
- Homomorphic Filtering
- Nonlinear Filtering (Denoising)
- Image Enhancement (Gamma Correction, CLAHE)
- Image Quality Metrics
- Image Resizing

Author: Refactored from original source files
"""

import os
import gc
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# GPU / CPU compatibility layer
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from cupyx.scipy.ndimage import (
        median_filter,
        maximum_filter,
        minimum_filter,
        uniform_filter,
        zoom,
    )
    HAS_CUPY = True
except ImportError:
    from scipy.ndimage import (
        median_filter,
        maximum_filter,
        minimum_filter,
        uniform_filter,
        zoom,
    )
    # Alias numpy as cp so existing class code works unchanged
    cp = np  # type: ignore[misc]
    HAS_CUPY = False

import matplotlib.pyplot as plt

from fabemd import FABEMD


def _to_numpy(arr) -> np.ndarray:
    """Convert cupy or numpy array to numpy."""
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return arr.get()
    return np.asarray(arr)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the image processing pipeline."""
    
    # Image paths
    proj_img_path: str = ""
    gain_img_path: str = ""
    dark_img_path: str = ""
    calibration_path: str = ""
    output_dir: str = ""
    
    # FFC parameters
    ffc_median_filter_size: int = 7
    
    # BEMD parameters
    bemd_max_iterations: int = 1
    bemd_threshold: float = 1.0
    bemd_initial_window_size: int = 32
    bemd_local_extrema_count: int = 10
    
    # Homomorphic filter parameter ranges
    d0_values: List[int] = field(default_factory=lambda: [20, 30, 40])
    rh_values: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    rl_values: List[float] = field(default_factory=lambda: [0.3, 0.5])
    
    # Gamma correction parameter ranges
    gamma_values: List[float] = field(default_factory=lambda: [0.8])
    
    # CLAHE parameter ranges
    clip_limit_values: List[float] = field(default_factory=lambda: [3.0])
    tile_grid_size_values: List[Tuple[int, int]] = field(default_factory=lambda: [(8, 8)])
    
    # Nonlinear filtering parameters
    denoise_r: int = 1
    denoise_beta: float = 0.5

    # Denoising method: "bilateral" (original) or "nlmeans" (PACE 2.0)
    denoise_method: str = "bilateral"

    # NL-means parameters (used when denoise_method="nlmeans")
    nlmeans_h: float = 10.0
    nlmeans_template_window: int = 7
    nlmeans_search_window: int = 21
    
    # FABEMD parameters (used when decomposition_method="fabemd")
    fabemd_max_sift_iterations: int = 10
    fabemd_sd_threshold: float = 0.2
    fabemd_min_extrema: int = 5
    fabemd_max_bimfs: int = 100
    fabemd_window_size_cap: int = 201
    fabemd_extrema_window: int = 3
    fabemd_initial_window_size: Optional[int] = None

    # Decomposition method: "bemd" or "fabemd"
    decomposition_method: str = "fabemd"

    # Homomorphic filter method: "gaussian" (classic) or "butterworth" (PACE)
    homomorphic_method: str = "gaussian"

    # PACE Butterworth filter parameters
    butterworth_order: int = 2
    pace_d0_values: List[int] = field(default_factory=lambda: [20, 30, 40])
    pace_gamma_h_values: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    pace_gamma_l_values: List[float] = field(default_factory=lambda: [0.3, 0.5])
    pace_n_values: List[int] = field(default_factory=lambda: [1, 2])

    # Output parameters
    output_width: int = 4096
    num_threads: int = 8

    # Processing mode: "full" (with FFC + spatial calibration) or "pace" (skip both)
    processing_mode: str = "full"
    
    @classmethod
    def from_json(cls, filepath: str) -> "PipelineConfig":
        """Load configuration from a JSON file."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        return cls(**{k: v for k, v in config_data.items() if k in cls.__dataclass_fields__})
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)


@dataclass
class ProcessingResult:
    """Result of image processing with evaluation metrics."""
    
    image: np.ndarray
    cii: float = 0.0
    entropy: float = 0.0
    eme: float = 0.0
    parameters: Optional[Tuple] = None
    
    @property
    def total_score(self) -> float:
        """Calculate total evaluation score."""
        return self.cii + self.entropy + self.eme


# =============================================================================
# Image Processing Modules
# =============================================================================

class FlatFieldCorrection:
    """Flat Field Correction (FFC) module."""
    
    def __init__(self, median_filter_size: int = 7):
        """
        Initialize FFC module.
        
        Args:
            median_filter_size: Size of median filter for noise reduction.
        """
        self.median_filter_size = median_filter_size
    
    def apply(
        self,
        projection_image: np.ndarray,
        gain_image: np.ndarray,
        dark_image: np.ndarray
    ) -> cp.ndarray:
        """
        Apply flat field correction to projection image.
        
        Args:
            projection_image: Raw projection image.
            gain_image: Gain calibration image.
            dark_image: Dark calibration image.
            
        Returns:
            Corrected image as CuPy array.
        """
        logger.info("Applying Flat Field Correction...")
        
        # Normalize and filter projection image
        proj = cp.asarray(projection_image)
        proj = (proj - cp.min(proj)) / (65535 - cp.min(proj))
        proj = median_filter(proj, self.median_filter_size)
        
        # Normalize and filter gain image
        gain = cp.asarray(gain_image)
        gain = (gain - cp.min(gain)) / (65535 - cp.min(gain))
        gain = median_filter(gain, self.median_filter_size)
        
        # Normalize and filter dark image
        dark = cp.asarray(dark_image)
        dark = (dark - cp.min(dark)) / (65535 - cp.min(dark))
        dark = median_filter(dark, self.median_filter_size)
        
        # Calculate corrected image
        proj_dark = cp.subtract(proj, dark)
        gain_dark = cp.subtract(gain, dark)
        
        proj_dark[proj_dark <= 0] = 0
        gain_dark[gain_dark <= 0] = 1e-12
        
        intensity = cp.divide(proj_dark, gain_dark)
        intensity[intensity <= 0] = 1e-12
        
        miu = -cp.log(intensity)
        miu = miu.astype(cp.float32)
        miu[miu < 0] = 0
        
        # Cleanup
        del proj, gain, dark, proj_dark, gain_dark, intensity
        gc.collect()
        
        logger.info("Flat Field Correction completed.")
        return miu


class SpatialCalibration:
    """Spatial calibration and distortion correction module."""
    
    def __init__(self, calibration_path: str):
        """
        Initialize spatial calibration module.
        
        Args:
            calibration_path: Path to calibration parameters file (.npz).
        """
        self.calibration_path = calibration_path
        self._load_parameters()
    
    def _load_parameters(self) -> None:
        """Load calibration parameters from file."""
        with np.load(self.calibration_path) as params:
            self.mtx = params['mtx']
            self.dist = params['dist']
            self.rvecs = params['rvecs']
            self.tvecs = params['tvecs']
            self.roi = params['roi']
    
    def _undistort(self, image: np.ndarray) -> np.ndarray:
        """Apply undistortion to image."""
        h, w = image.shape[:2]
        newcameramtx, roi_rotate = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
        return self._crop_image(undistorted, roi_rotate)
    
    def _crop_image(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop image to region of interest."""
        x, y, w, h = roi
        return image[y:y+h, x:x+w]
    
    def apply(self, image: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        """
        Apply spatial calibration to image.
        
        Args:
            image: Input image (numpy or cupy array).
            
        Returns:
            Calibrated and cropped image.
        """
        logger.info("Applying Spatial Calibration...")
        
        img = _to_numpy(image)
        
        img = self._undistort(img)
        img = self._crop_image(img, self.roi)
        
        logger.info("Spatial Calibration completed.")
        return img


class BEMD:
    """Bidimensional Empirical Mode Decomposition (BEMD) module."""
    
    def __init__(
        self,
        max_iterations: int = 10,
        threshold: float = 0.2,
        initial_window_size: int = 3,
        local_extrema_count: int = 5
    ):
        """
        Initialize BEMD module.
        
        Args:
            max_iterations: Maximum iterations for BIMF extraction.
            threshold: Threshold for standard deviation to accept BIMF.
            initial_window_size: Initial window size for finding extrema.
            local_extrema_count: Minimum extrema count stopping criteria.
        """
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.initial_window_size = initial_window_size
        self.local_extrema_count = local_extrema_count
    
    def _get_local_extrema(
        self, image: cp.ndarray, window_size: int = 3
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Get local maxima and minima maps."""
        mask = image != 0
        max_map = (image == maximum_filter(image, size=window_size)) & mask
        min_map = (image == minimum_filter(image, size=window_size)) & mask
        return max_map, min_map
    
    def _apply_order_statistic_filter(
        self,
        image: cp.ndarray,
        extrema_map: cp.ndarray,
        filter_type: str = "max",
        window_size: int = 3
    ) -> cp.ndarray:
        """Apply order statistics filter (MAX/MIN)."""
        if filter_type == "max":
            envelope = maximum_filter(image, size=window_size)
        elif filter_type == "min":
            envelope = minimum_filter(image, size=window_size)
        else:
            raise ValueError("filter_type should be either 'max' or 'min'")
        return cp.where(extrema_map, envelope, image)
    
    def _smooth_envelope(self, envelope: cp.ndarray, window_size: int = 3) -> cp.ndarray:
        """Smooth envelope with averaging filter."""
        return uniform_filter(envelope, size=window_size)
    
    def _calculate_mean_envelope(
        self, upper: cp.ndarray, lower: cp.ndarray
    ) -> cp.ndarray:
        """Calculate mean envelope."""
        return (upper + lower) / 2
    
    def _calculate_standard_deviation(
        self, FTj: cp.ndarray, FTj_next: cp.ndarray
    ) -> float:
        """Calculate standard deviation for BIMF criteria."""
        return float(cp.sqrt(cp.sum((FTj_next - FTj) ** 2) / cp.sum(FTj ** 2)))
    
    def decompose(self, image: cp.ndarray) -> List[cp.ndarray]:
        """
        Perform FABEMD decomposition on input image.
        
        Args:
            image: Input image as CuPy array.
            
        Returns:
            List of extracted BIMFs.
        """
        logger.info("Starting BEMD decomposition...")
        
        residual = image.astype(cp.float64)
        BIMFs = []
        limit_sd = self.threshold
        
        while True:
            FTj = residual.copy()
            window_size = self.initial_window_size
            SD = 1.0
            
            for j in range(self.max_iterations):
                # Find local extrema
                max_map, min_map = self._get_local_extrema(FTj, window_size)
                
                # Estimate envelopes
                upper_envelope = self._apply_order_statistic_filter(
                    FTj, max_map, "max", window_size
                )
                lower_envelope = self._apply_order_statistic_filter(
                    FTj, min_map, "min", window_size
                )
                
                # Smooth envelopes
                upper_envelope = self._smooth_envelope(upper_envelope, window_size)
                lower_envelope = self._smooth_envelope(lower_envelope, window_size)
                
                # Calculate mean envelope
                mean_envelope = self._calculate_mean_envelope(upper_envelope, lower_envelope)
                
                # Update FTj
                FTj_next = FTj - mean_envelope
                SD = self._calculate_standard_deviation(FTj, FTj_next)
                
                if SD < limit_sd:
                    BIMFs.append(FTj_next)
                    residual -= FTj_next
                    break
                else:
                    limit_sd = 1.1 * limit_sd
                    FTj = FTj_next
            
            # Check stopping criteria
            max_map, min_map = self._get_local_extrema(residual)
            extrema_count = int(cp.sum(max_map) + cp.sum(min_map))
            print(f"BEMD: {len(BIMFs)} BIMFs extracted, {extrema_count} extrema remaining", end="\r")
            
            if extrema_count <= self.local_extrema_count:
                logger.info(f"Stopping: fewer than {self.local_extrema_count} extrema points")
                break
            elif len(BIMFs) >= 100:
                logger.info("Stopping: 100 BIMFs extracted")
                break
            
            del max_map, min_map
            gc.collect()
        
        logger.info(f"BEMD decomposition completed. Extracted {len(BIMFs)} BIMFs.")
        return BIMFs
    
    @staticmethod
    def calculate_energies(bimfs: List[cp.ndarray]) -> List[float]:
        """Calculate energies of BIMFs."""
        energies = []
        for bimf in bimfs:
            energy = float(np.sum(np.square(_to_numpy(bimf))))
            energies.append(energy)
        return energies


class HomomorphicFilter:
    """Homomorphic filtering module for image enhancement."""
    
    def __init__(self, d0: float = 30, rh: float = 2.0, rl: float = 0.5, c: float = 1.0):
        """
        Initialize homomorphic filter.
        
        Args:
            d0: Cutoff frequency.
            rh: High frequency gain.
            rl: Low frequency gain.
            c: Filter sharpness constant.
        """
        self.d0 = d0
        self.rh = rh
        self.rl = rl
        self.c = c
    
    def apply(self, image: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        """
        Apply homomorphic filter to image.

        Pipeline:  image → ln → FFT → H(u,v) → IFFT → exp → result

        Args:
            image: Input image.
            
        Returns:
            Filtered image as float64 with corrected illumination.
        """
        if not isinstance(image, np.ndarray):
            img = _to_numpy(image)
        else:
            img = image.copy()
        img = img.astype(np.float64)
        rows, cols = img.shape

        # Ensure strictly positive values for log
        img = np.maximum(img, 1e-10)
        
        # Logarithmic transform
        log_image = np.log(img)
        
        # Fourier transform (numpy complex FFT)
        F = np.fft.fft2(log_image)
        F_shift = np.fft.fftshift(F)
        
        # Create Gaussian high-frequency emphasis filter
        u = np.arange(cols) - cols / 2
        v = np.arange(rows) - rows / 2
        U, V = np.meshgrid(u, v)
        d = np.sqrt(U ** 2 + V ** 2)
        H = (self.rh - self.rl) * (1 - np.exp(-self.c * (d ** 2 / self.d0 ** 2))) + self.rl
        
        # Apply filter
        F_filtered = F_shift * H
        
        # Inverse FFT — take real part
        filtered_log = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
        
        # Exponential transform (back from log domain)
        result = np.exp(filtered_log)

        # Normalise to [0, 1]
        rmin, rmax = result.min(), result.max()
        if rmax - rmin > 1e-10:
            result = (result - rmin) / (rmax - rmin)
        else:
            result = np.zeros_like(result)
        
        # Cleanup
        del img, log_image, F, F_shift, u, v, U, V, d, H, F_filtered, filtered_log
        gc.collect()
        
        return result


class PACEHomomorphicFilter:
    """Homomorphic filter using Butterworth high-pass (PACE, Siracusano et al. 2020).

    Unlike the Gaussian-based ``HomomorphicFilter``, this variant employs a
    Butterworth high-pass transfer function which provides a sharper frequency
    transition controlled by the filter order *n*:

        H(u,v) = (γH − γL) × ────────────────────── + γL
                                1 + (D0 / D(u,v))^(2n)

    where D(u,v) is the distance from the frequency centre.

    Reference
    ---------
    Siracusano, G. et al. "Pipeline for Advanced Contrast Enhancement (PACE)
    of Chest X-ray in Evaluating COVID-19 Patients", J. Digit. Imaging, 2020.
    """

    def __init__(
        self,
        d0: float = 30,
        gamma_h: float = 2.0,
        gamma_l: float = 0.5,
        n: int = 2,
    ):
        """
        Initialise PACE homomorphic filter.

        Args:
            d0: Cutoff frequency of the Butterworth high-pass filter.
            gamma_h: High-frequency gain (γH > 1 amplifies details).
            gamma_l: Low-frequency gain (γL < 1 compresses illumination).
            n: Order of the Butterworth filter (higher → sharper roll-off).
        """
        self.d0 = d0
        self.gamma_h = gamma_h
        self.gamma_l = gamma_l
        self.n = n

    def _butterworth_hpf(self, rows: int, cols: int) -> np.ndarray:
        """Build a Butterworth high-pass filter matrix of shape (rows, cols)."""
        u = np.arange(cols) - cols / 2
        v = np.arange(rows) - rows / 2
        U, V = np.meshgrid(u, v)
        D = np.sqrt(U ** 2 + V ** 2)
        # Avoid division by zero at DC
        D[D == 0] = 1e-12
        # Butterworth HPF: H_hp = 1 / (1 + (D0/D)^(2n))
        H_hp = 1.0 / (1.0 + (self.d0 / D) ** (2 * self.n))
        # Scale to [γL, γH]
        H = (self.gamma_h - self.gamma_l) * H_hp + self.gamma_l
        return H

    def apply(self, image: Union[np.ndarray, "cp.ndarray"]) -> np.ndarray:
        """Apply PACE homomorphic filter to a 2-D image.

        The classic homomorphic pipeline:
            image  →  ln  →  FFT  →  H(u,v)  →  IFFT  →  exp  →  result

        In the log domain, illumination (low-freq) and reflectance (high-freq)
        are additive.  The Butterworth HPF attenuates illumination (×γL < 1)
        while preserving / boosting reflectance (×γH ≥ 1), producing
        **homogeneous lighting** on the residual (see PACE paper, Fig. 3).

        Args:
            image: Input image (numpy or cupy array, any numeric dtype).

        Returns:
            Filtered image as float64 with corrected illumination.
        """
        img = _to_numpy(image).astype(np.float64)
        rows, cols = img.shape

        # 1. Ensure strictly positive values for log transform
        img = np.maximum(img, 1e-10)

        # 2. Logarithmic transform  (separates illumination × reflectance)
        log_image = np.log(img)

        # 3. FFT (numpy handles complex numbers natively)
        F = np.fft.fft2(log_image)
        F_shift = np.fft.fftshift(F)

        # 4. Build Butterworth high-pass filter and apply
        H = self._butterworth_hpf(rows, cols)
        F_filtered = F_shift * H

        # 5. Inverse FFT — take real part (imaginary ≈ 0 for real input)
        F_filtered = np.fft.ifftshift(F_filtered)
        filtered_log = np.real(np.fft.ifft2(F_filtered))

        # 6. Exponential transform  (back from log domain)
        result = np.exp(filtered_log)

        # 7. Normalise to [0, 1] for downstream compatibility
        rmin, rmax = result.min(), result.max()
        if rmax - rmin > 1e-10:
            result = (result - rmin) / (rmax - rmin)
        else:
            result = np.zeros_like(result)

        # Cleanup
        del img, log_image, F, F_shift, H, F_filtered, filtered_log
        gc.collect()

        return result


class NonlinearFilter:
    """Nonlinear filtering and denoising module."""
    
    def __init__(self, r: int = 1, beta: float = 0.5):
        """
        Initialize nonlinear filter.
        
        Args:
            r: Number of lowest-energy BIMFs to denoise.
            beta: Weight for filtered residual in reconstruction.
        """
        self.r = r
        self.beta = beta
    
    def denoise(
        self,
        bimfs: List[cp.ndarray],
        energies: List[float],
        filtered_residual: np.ndarray
    ) -> np.ndarray:
        """
        Denoise and reconstruct image from BIMFs.
        
        Args:
            bimfs: List of BIMFs from BEMD.
            energies: Energy values for each BIMF.
            filtered_residual: Filtered residual image.
            
        Returns:
            Reconstructed image.
        """
        # Sort BIMFs by energy
        sorted_indices = np.argsort(energies)
        
        # Denoise R components with lowest energy
        denoised_bimfs = []
        for i in range(int(self.r)):
            index = sorted_indices[i]
            denoised = cv2.bilateralFilter(
                _to_numpy(bimfs[index]).astype(np.float32), 5, 75, 75
            )
            denoised_bimfs.append(denoised)
        
        # Combine denoised and original BIMFs
        I_E = np.sum(denoised_bimfs, axis=0)
        for j in range(int(self.r), len(bimfs)):
            index = sorted_indices[j]
            I_E += _to_numpy(bimfs[index])
        
        # Reconstruct with filtered residual
        I_L = I_E + self.beta * filtered_residual
        return I_L


class NLMeansFilter:
    """Non-Local Means denoising module (PACE 2.0, Siracusano et al. 2023).

    Replaces bilateral filtering with NL-means, which exploits self-similarity
    across the *entire* image rather than only local neighbourhoods.  For each
    pixel the algorithm averages all pixels whose surrounding patch is similar,
    weighted by the Gaussian-weighted patch distance:

        NL[v](i) = Σ_j  w(i,j) · v(j)
        w(i,j)   = (1/Z(i)) · exp( −‖v(N_i) − v(N_j)‖²_{2,a} / h² )

    where *h* controls the degree of filtering.  This is particularly effective
    on medical radiographs where repetitive anatomical textures exist.

    Reference
    ---------
    Siracusano, G. et al. "Effective processing pipeline PACE 2.0 for
    enhancing chest x-ray contrast and diagnostic interpretability",
    Scientific Reports, 2023.
    """

    def __init__(
        self,
        r: int = 1,
        beta: float = 0.5,
        h: float = 10.0,
        template_window_size: int = 7,
        search_window_size: int = 21,
    ):
        """
        Initialise NL-means filter.

        Args:
            r: Number of lowest-energy BIMFs to denoise.
            beta: Weight for filtered residual in reconstruction.
            h: Filter strength (higher removes more noise but may blur).
            template_window_size: Size of the patch used for comparison (odd).
            search_window_size: Size of the area searched for similar patches (odd).
        """
        self.r = r
        self.beta = beta
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def _nlmeans_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply NL-means denoising to a single-channel image.

        OpenCV's ``fastNlMeansDenoising`` with NORM_L2 only supports uint8.
        We normalise to uint8, denoise, then rescale back to the original
        range so the filter integrates transparently with the rest of the
        pipeline.
        """
        img = image.astype(np.float32)
        vmin, vmax = img.min(), img.max()
        denom = vmax - vmin if vmax != vmin else 1.0

        # Scale to uint8 for cv2.fastNlMeansDenoising (NORM_L2 requirement)
        img_u8 = np.uint8(np.clip((img - vmin) / denom * 255, 0, 255))

        denoised_u8 = cv2.fastNlMeansDenoising(
            img_u8,
            None,
            h=self.h,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size,
        )

        # Scale back to original range
        denoised = denoised_u8.astype(np.float32) / 255.0 * denom + vmin
        return denoised

    def denoise(
        self,
        bimfs: List,
        energies: List[float],
        filtered_residual: np.ndarray,
    ) -> np.ndarray:
        """
        Denoise and reconstruct image from BIMFs using NL-means.

        The interface is identical to ``NonlinearFilter.denoise`` so both
        classes are drop-in replaceable.

        Args:
            bimfs: List of BIMFs from BEMD / FABEMD.
            energies: Energy values for each BIMF.
            filtered_residual: Homomorphic-filtered residual image.

        Returns:
            Reconstructed image (float32/64 numpy array).
        """
        # Sort BIMFs by energy (lowest first)
        sorted_indices = np.argsort(energies)

        # Denoise the R lowest-energy BIMFs with NL-means
        denoised_bimfs = []
        for i in range(int(self.r)):
            index = sorted_indices[i]
            bimf_np = _to_numpy(bimfs[index]).astype(np.float32)
            denoised = self._nlmeans_denoise(bimf_np)
            denoised_bimfs.append(denoised)

        # Sum denoised + remaining original BIMFs
        I_E = np.sum(denoised_bimfs, axis=0)
        for j in range(int(self.r), len(bimfs)):
            index = sorted_indices[j]
            I_E += _to_numpy(bimfs[index])

        # Reconstruct with filtered residual
        I_L = I_E + self.beta * filtered_residual
        return I_L


class ImageEnhancer:
    """Image enhancement module (Gamma correction, CLAHE)."""
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
        """
        Apply gamma correction.
        
        Args:
            image: Input image (float or uint16).
            gamma: Gamma value (< 1 brightens, > 1 darkens).
            
        Returns:
            Gamma-corrected image as uint16.
        """
        img = image.astype(np.float64)
        # Normalise to [0, 1] regardless of input range/dtype
        imin, imax = img.min(), img.max()
        if imax - imin > 1e-10:
            img_normalized = (img - imin) / (imax - imin)
        else:
            img_normalized = np.zeros_like(img)
        img_corrected = np.power(np.clip(img_normalized, 0, 1), gamma)
        return np.uint16(img_corrected * 65535)
    
    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 0.5,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input image (uint16).
            clip_limit: Threshold for contrast limiting.
            tile_grid_size: Size of grid for histogram equalization.
            
        Returns:
            CLAHE-enhanced image.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


class ImageMetrics:
    """Image quality metrics calculation module."""
    
    @staticmethod
    def calculate_contrast(image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate contrast of region defined by mask."""
        foreground = image[mask == 1]
        background = image[mask == 0]
        
        X_f = np.mean(foreground) if len(foreground) > 0 else 0
        X_b = np.mean(background) if len(background) > 0 else 0
        
        if X_f + X_b == 0:
            return 0.0
        
        return (X_f - X_b) / (X_f + X_b)
    
    @staticmethod
    def calculate_cii(
        processed: np.ndarray,
        reference: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """Calculate Contrast Improvement Index."""
        C_processed = ImageMetrics.calculate_contrast(processed, mask)
        C_reference = ImageMetrics.calculate_contrast(reference, mask)
        
        if C_reference == 0:
            return 0.0
        
        return C_processed / C_reference
    
    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """Calculate image entropy."""
        hist = cv2.calcHist([image], [0], None, [65535], [0, 65535])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-7))
        return float(entropy)
    
    @staticmethod
    def calculate_eme(
        image: np.ndarray,
        r: int,
        c: int,
        epsilon: float = 0.0001
    ) -> float:
        """
        Calculate Effective Measure of Enhancement.
        
        Args:
            image: Input image.
            r: Number of row blocks.
            c: Number of column blocks.
            epsilon: Small constant to avoid division by zero.
            
        Returns:
            EME value.
        """
        height, width = image.shape
        block_height = height // r
        block_width = width // c
        
        eme = 0.0
        for i in range(r):
            for j in range(c):
                block = image[
                    i * block_height:(i + 1) * block_height,
                    j * block_width:(j + 1) * block_width,
                ]
                
                I_max = np.max(block)
                I_min = np.min(block)
                
                if I_min + epsilon == 0:
                    continue
                
                CR = I_max / (I_min + epsilon)
                eme += 20 * np.log(CR)
        
        return eme / (r * c)


class ImageResizer:
    """Image resizing module using GPU acceleration."""
    
    @staticmethod
    def resize(image: Union[np.ndarray, cp.ndarray], new_width: int) -> cp.ndarray:
        """
        Resize image to specified width maintaining aspect ratio.
        
        Args:
            image: Input image.
            new_width: Target width.
            
        Returns:
            Resized image as CuPy array.
        """
        if isinstance(image, np.ndarray):
            image = cp.array(image)
        
        height, width = image.shape[:2]
        width_percent = new_width / float(width)
        new_height = int(height * width_percent)
        
        if len(image.shape) == 3:
            zoom_factors = (new_height / height, new_width / width, 1)
        else:
            zoom_factors = (new_height / height, new_width / width)
        
        return zoom(image, zoom_factors, order=1)


# =============================================================================
# Main Pipeline Class
# =============================================================================

class ImageProcessingPipeline:
    """
    Main image processing pipeline that orchestrates all processing modules.
    
    This class provides a unified interface for medical image processing,
    including flat field correction, calibration, enhancement, and optimization.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the image processing pipeline.
        
        Args:
            config: Pipeline configuration. If None, uses default values.
        """
        self.config = config or PipelineConfig()
        self._init_modules()
    
    def _init_modules(self) -> None:
        """Initialize processing modules with current configuration."""
        self.ffc = FlatFieldCorrection(self.config.ffc_median_filter_size)
        self.bemd = BEMD(
            max_iterations=self.config.bemd_max_iterations,
            threshold=self.config.bemd_threshold,
            initial_window_size=self.config.bemd_initial_window_size,
            local_extrema_count=self.config.bemd_local_extrema_count,
        )
        self.fabemd = FABEMD(
            max_sift_iterations=self.config.fabemd_max_sift_iterations,
            sd_threshold=self.config.fabemd_sd_threshold,
            min_extrema=self.config.fabemd_min_extrema,
            max_bimfs=self.config.fabemd_max_bimfs,
            initial_window_size=self.config.fabemd_initial_window_size,
            window_size_cap=self.config.fabemd_window_size_cap,
            extrema_window=self.config.fabemd_extrema_window,
        )
        denoise_method = self.config.denoise_method.lower().strip()
        if denoise_method == "nlmeans":
            self.nonlinear_filter = NLMeansFilter(
                r=self.config.denoise_r,
                beta=self.config.denoise_beta,
                h=self.config.nlmeans_h,
                template_window_size=self.config.nlmeans_template_window,
                search_window_size=self.config.nlmeans_search_window,
            )
        else:
            self.nonlinear_filter = NonlinearFilter(
                r=self.config.denoise_r,
                beta=self.config.denoise_beta,
            )
        self.enhancer = ImageEnhancer()
        self.metrics = ImageMetrics()
        self.resizer = ImageResizer()
    
    def load_images(
        self,
        proj_path: Optional[str] = None,
        gain_path: Optional[str] = None,
        dark_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load projection, gain, and dark images.
        
        Args:
            proj_path: Path to projection image. Uses config if None.
            gain_path: Path to gain image. Uses config if None.
            dark_path: Path to dark image. Uses config if None.
            
        Returns:
            Tuple of (projection, gain, dark) images.
        """
        logger.info("Loading images...")
        
        proj_path = proj_path or self.config.proj_img_path
        gain_path = gain_path or self.config.gain_img_path
        dark_path = dark_path or self.config.dark_img_path
        
        proj_img = cv2.imread(proj_path, -1)
        gain_img = cv2.imread(gain_path, -1)
        dark_img = cv2.imread(dark_path, -1)
        
        if proj_img is None:
            raise FileNotFoundError(f"Could not load projection image: {proj_path}")
        if gain_img is None:
            raise FileNotFoundError(f"Could not load gain image: {gain_path}")
        if dark_img is None:
            raise FileNotFoundError(f"Could not load dark image: {dark_path}")
        
        logger.info("Images loaded successfully.")
        return proj_img, gain_img, dark_img

    def load_projection(self, proj_path: Optional[str] = None) -> np.ndarray:
        """
        Load projection image only (PACE mode).

        Args:
            proj_path: Path to projection image. Uses config if None.

        Returns:
            Projection image.
        """
        logger.info("Loading projection image...")

        proj_path = proj_path or self.config.proj_img_path
        proj_img = cv2.imread(proj_path, -1)

        if proj_img is None:
            raise FileNotFoundError(f"Could not load projection image: {proj_path}")

        logger.info("Projection image loaded successfully.")
        return proj_img
    
    def apply_ffc(
        self,
        proj_img: np.ndarray,
        gain_img: np.ndarray,
        dark_img: np.ndarray
    ) -> cp.ndarray:
        """Apply flat field correction."""
        return self.ffc.apply(proj_img, gain_img, dark_img)
    
    def apply_spatial_calibration(
        self,
        image: Union[np.ndarray, cp.ndarray],
        calibration_path: Optional[str] = None
    ) -> np.ndarray:
        """Apply spatial calibration to image."""
        calib_path = calibration_path or self.config.calibration_path
        calibrator = SpatialCalibration(calib_path)
        return calibrator.apply(image)
    
    def decompose_image(
        self, image: np.ndarray, method: Optional[str] = None
    ) -> Tuple[List[cp.ndarray], List[float]]:
        """
        Decompose image using BEMD or FABEMD.

        Args:
            image: Input image (numpy array).
            method: "bemd" or "fabemd". Defaults to config.decomposition_method.

        Returns:
            Tuple of (BIMFs, energies).
        """
        method = (method or self.config.decomposition_method).lower().strip()
        gpu_image = cp.asarray(image)

        if method == "fabemd":
            logger.info("Starting image decomposition (FABEMD)...")
            bimfs = self.fabemd.decompose(gpu_image)
            energies = FABEMD.calculate_energies(bimfs)
        else:
            logger.info("Starting image decomposition (BEMD)...")
            bimfs = self.bemd.decompose(gpu_image)
            energies = BEMD.calculate_energies(bimfs)

        logger.info("Image decomposition completed.")
        return bimfs, energies
    
    def _process_single_params(
        self,
        params: Tuple,
        reference_image: np.ndarray,
        bimfs: List[cp.ndarray],
        energies: List[float]
    ) -> ProcessingResult:
        """Process image with single parameter combination."""
        hom_method = self.config.homomorphic_method.lower().strip()

        if hom_method == "butterworth":
            d0, gamma_h, gamma_l, bw_n, gamma, clip_limit, tile_grid_size = params
            hf = PACEHomomorphicFilter(d0=d0, gamma_h=gamma_h, gamma_l=gamma_l, n=bw_n)
        else:
            d0, rh, rl, gamma, clip_limit, tile_grid_size = params
            hf = HomomorphicFilter(d0=d0, rh=rh, rl=rl)

        # Apply homomorphic filter
        filtered_image = hf.apply(reference_image)
        
        # Reconstruct image
        reconstructed = self.nonlinear_filter.denoise(bimfs, energies, filtered_image)
        
        # Apply gamma correction
        gamma_corrected = self.enhancer.gamma_correction(reconstructed, gamma)
        
        # Apply CLAHE
        clahe_image = self.enhancer.apply_clahe(gamma_corrected, clip_limit, tile_grid_size)
        
        # Calculate metrics
        mask = np.ones_like(reference_image)
        cii = self.metrics.calculate_cii(clahe_image, reference_image, mask)
        entropy = self.metrics.calculate_entropy(clahe_image)
        eme = self.metrics.calculate_eme(clahe_image, 4, 4)
        
        # Cleanup
        del filtered_image, reconstructed, gamma_corrected, mask
        gc.collect()
        
        return ProcessingResult(
            image=clahe_image,
            cii=cii,
            entropy=entropy,
            eme=eme,
            parameters=params
        )
    
    def find_best_parameters(
        self,
        reference_image: np.ndarray,
        bimfs: List[cp.ndarray],
        energies: List[float]
    ) -> ProcessingResult:
        """
        Find best processing parameters through grid search.
        
        Args:
            reference_image: Reference image for processing.
            bimfs: BIMFs from BEMD decomposition.
            energies: Energy values for BIMFs.
            
        Returns:
            ProcessingResult with best parameters and image.
        """
        logger.info("Finding best parameters...")
        
        # Generate parameter combinations
        hom_method = self.config.homomorphic_method.lower().strip()
        if hom_method == "butterworth":
            parameter_combinations = list(itertools.product(
                self.config.pace_d0_values,
                self.config.pace_gamma_h_values,
                self.config.pace_gamma_l_values,
                self.config.pace_n_values,
                self.config.gamma_values,
                self.config.clip_limit_values,
                self.config.tile_grid_size_values,
            ))
        else:
            parameter_combinations = list(itertools.product(
                self.config.d0_values,
                self.config.rh_values,
                self.config.rl_values,
                self.config.gamma_values,
                self.config.clip_limit_values,
                self.config.tile_grid_size_values,
            ))
        
        logger.info(f"Total parameter combinations: {len(parameter_combinations)}")
        
        best_result = ProcessingResult(image=np.array([]))
        
        with ThreadPoolExecutor(self.config.num_threads) as executor:
            futures = [
                executor.submit(
                    self._process_single_params,
                    params, reference_image, bimfs, energies
                )
                for params in parameter_combinations
            ]
            
            for future in as_completed(futures):
                result = future.result()
                logger.debug(
                    f"Params {result.parameters}, Score: {result.total_score:.4f}, "
                    f"CII: {result.cii:.4f}, Entropy: {result.entropy:.4f}, EME: {result.eme:.4f}"
                )
                
                if result.total_score > best_result.total_score:
                    best_result = result
        
        logger.info(f"Best parameters found: {best_result.parameters}")
        return best_result
    
    def normalize_and_resize(
        self,
        image: np.ndarray,
        target_width: Optional[int] = None
    ) -> np.ndarray:
        """Normalize image to 16-bit and resize."""
        target_width = target_width or self.config.output_width
        
        # Convert to CuPy for GPU processing
        img = cp.asarray(image)
        
        # Normalize to 16-bit range
        min_val = cp.min(img)
        max_val = cp.max(img)
        img = min_val + ((img - min_val) / (max_val - min_val) * 65535)
        
        # Resize
        resized = self.resizer.resize(img, target_width)
        
        return _to_numpy(resized)
    
    def save_image(
        self,
        image: np.ndarray,
        output_path: str,
        compression: int = 1
    ) -> str:
        """
        Save image to file.
        
        Args:
            image: Image to save.
            output_path: Output file path.
            compression: TIFF compression level.
            
        Returns:
            Path to saved file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(
            output_path,
            image.astype(np.uint16),
            params=(cv2.IMWRITE_TIFF_COMPRESSION, compression)
        )
        logger.info(f"Image saved to: {output_path}")
        return output_path
    
    def process(
        self,
        proj_path: Optional[str] = None,
        gain_path: Optional[str] = None,
        dark_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        output_path: Optional[str] = None,
        show_plot: bool = True,
        mode: Optional[str] = None
    ) -> ProcessingResult:
        """
        Run the complete image processing pipeline.
        
        Args:
            proj_path: Path to projection image.
            gain_path: Path to gain image.
            dark_path: Path to dark image.
            calibration_path: Path to calibration file.
            output_path: Path for output image.
            show_plot: Whether to display result plot.
            
        Returns:
            ProcessingResult with best processed image.
        """
        try:
            processing_mode = (mode or self.config.processing_mode).lower().strip()

            if processing_mode not in {"full", "pace"}:
                raise ValueError("mode must be either 'full' or 'pace'")

            if processing_mode == "pace":
                # Load projection only
                proj_img = self.load_projection(proj_path)
                calibrated_img = proj_img
                ffc_img = None
                gain_img = None
                dark_img = None
            else:
                # Load images
                proj_img, gain_img, dark_img = self.load_images(proj_path, gain_path, dark_path)

                # Apply flat field correction
                ffc_img = self.apply_ffc(proj_img, gain_img, dark_img)

                # Apply spatial calibration
                calibrated_img = self.apply_spatial_calibration(ffc_img, calibration_path)
            
            # Decompose image
            bimfs, energies = self.decompose_image(calibrated_img)
            
            # Find best parameters
            best_result = self.find_best_parameters(calibrated_img, bimfs, energies)
            
            # Normalize and resize
            final_image = self.normalize_and_resize(best_result.image)
            best_result.image = final_image
            
            # Save result
            if output_path:
                self.save_image(final_image, output_path)
            elif self.config.output_dir:
                filename = Path(proj_path or self.config.proj_img_path).stem + "_processed.tiff"
                out_path = os.path.join(self.config.output_dir, filename)
                self.save_image(final_image, out_path)
            
            # Display results
            if show_plot:
                if processing_mode == "pace":
                    self._plot_results(proj_img, None, final_image)
                else:
                    self._plot_results(proj_img, calibrated_img, final_image)
            
            # Cleanup
            self._cleanup(
                proj_img, gain_img, dark_img, ffc_img, calibrated_img, bimfs, energies
            )
            
            logger.info("Pipeline completed successfully.")
            return best_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _plot_results(
        self,
        original: np.ndarray,
        calibrated: Optional[np.ndarray],
        processed: np.ndarray
    ) -> None:
        """Display processing results."""
        if calibrated is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            axes[1].imshow(processed, cmap='gray')
            axes[1].set_title('Processed Image')
            axes[1].axis('off')
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(calibrated, cmap='gray')
            axes[1].set_title('Calibrated Image')
            axes[1].axis('off')

            axes[2].imshow(processed, cmap='gray')
            axes[2].set_title('Processed Image')
            axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    
    def _cleanup(self, *arrays) -> None:
        """Clean up memory."""
        logger.info("Cleaning up memory...")
        for arr in arrays:
            if arr is not None:
                del arr
        if HAS_CUPY:
            cp._default_memory_pool.free_all_blocks()
        gc.collect()
        logger.info("Memory cleaned.")

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        mode: Optional[str] = None,
        extensions: Tuple[str, ...] = (".tiff", ".tif", ".mdn")
    ) -> List[ProcessingResult]:
        """
        Batch process images in a directory.

        Args:
            input_dir: Directory with projection images.
            output_dir: Directory for processed outputs.
            mode: "full" or "pace". Defaults to config.processing_mode.
            extensions: File extensions to include.

        Returns:
            List of ProcessingResult objects.
        """
        results: List[ProcessingResult] = []
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(extensions):
                proj_path = os.path.join(input_dir, filename)
                output_path = os.path.join(
                    output_dir,
                    Path(filename).stem + "_processed.tiff"
                )

                result = self.process(
                    proj_path=proj_path,
                    output_path=output_path,
                    show_plot=False,
                    mode=mode
                )
                results.append(result)

        return results


# =============================================================================
# Example Usage
# =============================================================================

def main():
    """Example usage of the image processing pipeline."""
    
    # Create configuration
    config = PipelineConfig(
        proj_img_path="path/to/projection.tiff",
        gain_img_path="path/to/gain.mdn",
        dark_img_path="path/to/dark.mdn",
        calibration_path="path/to/calibration.npz",
        output_dir="path/to/output",
        
        # BEMD parameters
        bemd_max_iterations=1,
        bemd_threshold=1.0,
        bemd_initial_window_size=32,
        bemd_local_extrema_count=10,
        
        # Parameter search ranges
        d0_values=[20, 30, 40],
        rh_values=[1.5, 2.0, 2.5],
        rl_values=[0.3, 0.5],
        gamma_values=[0.8],
        clip_limit_values=[3.0],
        tile_grid_size_values=[(8, 8)],
        
        # Output settings
        output_width=4096,
        num_threads=8,
    )
    
    # Or load from JSON
    # config = PipelineConfig.from_json("config.json")
    
    # Create pipeline
    pipeline = ImageProcessingPipeline(config)
    
    # Run processing
    result = pipeline.process(show_plot=True)
    
    print(f"Best parameters: {result.parameters}")
    print(f"CII: {result.cii:.4f}")
    print(f"Entropy: {result.entropy:.4f}")
    print(f"EME: {result.eme:.4f}")
    print(f"Total Score: {result.total_score:.4f}")


if __name__ == "__main__":
    main()
