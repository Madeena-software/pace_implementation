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
import cupy as cp
from cupyx.scipy.ndimage import (
    median_filter,
    maximum_filter,
    minimum_filter,
    uniform_filter,
    zoom,
)
import matplotlib.pyplot as plt


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
    
    # Output parameters
    output_width: int = 4096
    num_threads: int = 8
    
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
        
        proj_dark[proj_dark <= 0] = 1e-12
        gain_dark[gain_dark <= 0] = 0
        
        intensity = cp.divide(gain_dark, proj_dark)
        intensity[intensity <= 0] = 1e-12
        
        miu = cp.log(intensity)
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
        
        if isinstance(image, cp.ndarray):
            img = image.get()
        else:
            img = image
        
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
            energy = float(np.sum(np.square(np.array(bimf.get()))))
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
        
        Args:
            image: Input image.
            
        Returns:
            Filtered image as uint16.
        """
        if not isinstance(image, np.ndarray):
            img = image.get()
        else:
            img = image
        
        rows, cols = img.shape
        
        # Logarithmic transform
        log_image = np.log1p(img)
        
        # Fourier transform
        dft = cv2.dft(log_image, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create high-frequency emphasis filter
        u = np.arange(cols)
        v = np.arange(rows)
        u, v = np.meshgrid(u - rows / 2, v - cols / 2)
        d = np.sqrt(u ** 2 + v ** 2)
        h = (self.rh - self.rl) * (1 - np.exp(-self.c * (d ** 2 / self.d0 ** 2))) + self.rl
        h = np.repeat(h[:, :, np.newaxis], 2, axis=2)
        
        # Apply filter
        dft_shift_filtered = dft_shift * h
        
        # Inverse Fourier transform
        dft_shift_filtered = np.fft.ifftshift(dft_shift_filtered)
        idft = cv2.idft(dft_shift_filtered)
        idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
        
        # Normalize
        idft = cv2.normalize(idft, None, 0, 1, cv2.NORM_MINMAX)
        
        # Exponential transform
        exp_image = np.expm1(idft)
        exp_image = np.nan_to_num(exp_image, nan=0.0, posinf=65535.0, neginf=0.0)
        
        # Final normalization
        exp_image = cv2.normalize(exp_image, None, 0, 65535, cv2.NORM_MINMAX)
        exp_image = np.uint16(exp_image)
        
        # Cleanup
        del img, log_image, dft, dft_shift, u, v, d, h, dft_shift_filtered, idft
        gc.collect()
        
        return exp_image


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
                bimfs[index].get().astype(np.float32), 5, 75, 75
            )
            denoised_bimfs.append(denoised)
        
        # Combine denoised and original BIMFs
        I_E = np.sum(denoised_bimfs, axis=0)
        for j in range(int(self.r), len(bimfs)):
            index = sorted_indices[j]
            I_E += np.array(bimfs[index].get())
        
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
            image: Input image.
            gamma: Gamma value (< 1 brightens, > 1 darkens).
            
        Returns:
            Gamma-corrected image.
        """
        img_normalized = image / 65535.0
        img_corrected = np.power(img_normalized, gamma)
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
    
    def decompose_image(self, image: np.ndarray) -> Tuple[List[cp.ndarray], List[float]]:
        """
        Decompose image using BEMD.
        
        Returns:
            Tuple of (BIMFs, energies).
        """
        logger.info("Starting image decomposition (BEMD)...")
        bimfs = self.bemd.decompose(cp.asarray(image))
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
        d0, rh, rl, gamma, clip_limit, tile_grid_size = params
        
        # Apply homomorphic filter
        hf = HomomorphicFilter(d0=d0, rh=rh, rl=rl)
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
        
        return resized.get()
    
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
        show_plot: bool = True
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
        calibrated: np.ndarray,
        processed: np.ndarray
    ) -> None:
        """Display processing results."""
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
            del arr
        cp._default_memory_pool.free_all_blocks()
        gc.collect()
        logger.info("Memory cleaned.")


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
