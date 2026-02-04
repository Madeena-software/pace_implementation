"""
Image Processing Pipeline - Package Initialization
===================================================

A unified, class-based image processing pipeline for medical imaging.

Quick Start:
------------
    from image_pipeline import ImageProcessingPipeline, PipelineConfig
    
    # Create config
    config = PipelineConfig(
        proj_img_path="path/to/image.tiff",
        gain_img_path="path/to/gain.mdn",
        dark_img_path="path/to/dark.mdn",
        calibration_path="path/to/calibration.npz"
    )
    
    # Create pipeline and process
    pipeline = ImageProcessingPipeline(config)
    result = pipeline.process()

Available Classes:
------------------
- PipelineConfig: Configuration dataclass for the pipeline
- ProcessingResult: Result container with image and metrics
- ImageProcessingPipeline: Main orchestrator class

Processing Modules:
-------------------
- FlatFieldCorrection: Flat field correction using GPU
- SpatialCalibration: Lens distortion correction
- BEMD: Bidimensional Empirical Mode Decomposition
- HomomorphicFilter: Frequency domain filtering
- NonlinearFilter: Bilateral filtering and denoising
- ImageEnhancer: Gamma correction and CLAHE
- ImageMetrics: CII, entropy, and EME calculations
- ImageResizer: GPU-accelerated image resizing
"""

from .image_pipeline import (
    # Configuration
    PipelineConfig,
    ProcessingResult,
    
    # Main Pipeline
    ImageProcessingPipeline,
    
    # Processing Modules
    FlatFieldCorrection,
    SpatialCalibration,
    BEMD,
    HomomorphicFilter,
    NonlinearFilter,
    ImageEnhancer,
    ImageMetrics,
    ImageResizer,
)

__version__ = "1.0.0"
__author__ = "Refactored Pipeline"

__all__ = [
    # Configuration
    "PipelineConfig",
    "ProcessingResult",
    
    # Main Pipeline
    "ImageProcessingPipeline",
    
    # Processing Modules
    "FlatFieldCorrection",
    "SpatialCalibration",
    "BEMD",
    "HomomorphicFilter",
    "NonlinearFilter",
    "ImageEnhancer",
    "ImageMetrics",
    "ImageResizer",
]
