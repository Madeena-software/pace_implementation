# Image Processing Pipeline - Refactored

A unified, class-based image processing pipeline for medical imaging.

## Features

- **Modular Design**: Each processing step is encapsulated in its own class
- **Easy Configuration**: Use `PipelineConfig` dataclass or JSON files
- **GPU Acceleration**: Uses CuPy for GPU-accelerated processing
- **Parallel Processing**: ThreadPoolExecutor for parameter optimization
- **Type Hints**: Full type annotations for better IDE support
- **Logging**: Comprehensive logging throughout the pipeline

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy opencv-python cupy matplotlib
```

## Quick Start

### Using Python

```python
from image_pipeline import ImageProcessingPipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    proj_img_path="path/to/projection.tiff",
    gain_img_path="path/to/gain.mdn",
    dark_img_path="path/to/dark.mdn",
    calibration_path="path/to/calibration.npz",
    output_dir="path/to/output",
)

# Create pipeline and process
pipeline = ImageProcessingPipeline(config)
result = pipeline.process(show_plot=True)

print(f"Best parameters: {result.parameters}")
print(f"Total Score: {result.total_score:.4f}")
```

### PACE Mode (skip FFC + spatial calibration)

```python
from image_pipeline import ImageProcessingPipeline, PipelineConfig

config = PipelineConfig(
    proj_img_path="path/to/projection.tiff",
    output_dir="path/to/output",
    processing_mode="pace",
)

pipeline = ImageProcessingPipeline(config)
result = pipeline.process(show_plot=True, mode="pace")
```

### Using JSON Configuration

```python
from image_pipeline import ImageProcessingPipeline, PipelineConfig

# Load configuration from JSON
config = PipelineConfig.from_json("config.json")

# Create and run pipeline
pipeline = ImageProcessingPipeline(config)
result = pipeline.process()
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `proj_img_path` | str | "" | Path to projection image |
| `gain_img_path` | str | "" | Path to gain calibration image |
| `dark_img_path` | str | "" | Path to dark calibration image |
| `calibration_path` | str | "" | Path to spatial calibration file (.npz) |
| `output_dir` | str | "" | Output directory for processed images |
| `ffc_median_filter_size` | int | 7 | Median filter size for FFC |
| `bemd_max_iterations` | int | 1 | Max iterations for BEMD |
| `bemd_threshold` | float | 1.0 | BIMF acceptance threshold |
| `bemd_initial_window_size` | int | 32 | Initial window for extrema detection |
| `bemd_local_extrema_count` | int | 10 | Stopping criteria for extrema |
| `d0_values` | List[int] | [20, 30, 40] | Cutoff frequencies to try |
| `rh_values` | List[float] | [1.5, 2.0, 2.5] | High freq gains to try |
| `rl_values` | List[float] | [0.3, 0.5] | Low freq gains to try |
| `gamma_values` | List[float] | [0.8] | Gamma values to try |
| `clip_limit_values` | List[float] | [3.0] | CLAHE clip limits to try |
| `tile_grid_size_values` | List[Tuple] | [(8, 8)] | CLAHE grid sizes to try |
| `denoise_r` | int | 1 | Number of BIMFs to denoise |
| `denoise_beta` | float | 0.5 | Residual weight in reconstruction |
| `output_width` | int | 4096 | Target output width |
| `num_threads` | int | 8 | Number of parallel threads |
| `processing_mode` | str | "full" | "full" or "pace" |

## Processing Pipeline

1. **Load Images**: Load projection, gain, and dark images
2. **Flat Field Correction**: Apply FFC using GPU-accelerated median filtering
3. **Spatial Calibration**: Apply lens distortion correction
4. **BEMD Decomposition**: Extract BIMFs from image
5. **Parameter Optimization**: Find best processing parameters through grid search
   - Homomorphic filtering
   - Nonlinear filtering (bilateral filter denoising)
   - Gamma correction
   - CLAHE enhancement
6. **Evaluation**: Calculate CII, entropy, and EME metrics
7. **Output**: Normalize, resize, and save the result

### PACE Processing Flow

1. **Load Image**: Load projection image only
2. **BEMD Decomposition**: Extract BIMFs from image
3. **Parameter Optimization**: Homomorphic filtering, denoising, gamma correction, CLAHE
4. **Evaluation**: Calculate CII, entropy, and EME metrics
5. **Output**: Normalize, resize, and save the result

## Using Individual Modules

Each processing module can be used independently:

```python
from image_pipeline import (
    FlatFieldCorrection,
    BEMD,
    HomomorphicFilter,
    ImageEnhancer,
    ImageMetrics
)

# Flat Field Correction
ffc = FlatFieldCorrection(median_filter_size=7)
corrected = ffc.apply(proj_img, gain_img, dark_img)

# BEMD Decomposition
bemd = BEMD(max_iterations=1, threshold=1.0)
bimfs = bemd.decompose(cp.asarray(image))
energies = BEMD.calculate_energies(bimfs)

# Homomorphic Filter
hf = HomomorphicFilter(d0=30, rh=2.0, rl=0.5)
filtered = hf.apply(image)

# Image Enhancement
enhanced = ImageEnhancer.gamma_correction(image, gamma=0.8)
clahe_image = ImageEnhancer.apply_clahe(image, clip_limit=3.0)

# Metrics
cii = ImageMetrics.calculate_cii(processed, reference, mask)
entropy = ImageMetrics.calculate_entropy(image)
eme = ImageMetrics.calculate_eme(image, r=4, c=4)
```

## File Structure

```
refactored/
├── __init__.py          # Package initialization with exports
├── image_pipeline.py    # Main module with all classes
├── config.json          # Example configuration file
└── README.md            # This file
```

## Batch Processing (PACE)

```python
from image_pipeline import ImageProcessingPipeline, PipelineConfig

config = PipelineConfig(processing_mode="pace")
pipeline = ImageProcessingPipeline(config)

results = pipeline.process_batch(
    input_dir="path/to/projections",
    output_dir="path/to/output",
    mode="pace"
)
```

## Migration from Original Code

The refactored code consolidates all modules from the original `source/` directory:

| Original File | New Location |
|--------------|--------------|
| `ffc.py` | `FlatFieldCorrection` class |
| `calibrate_image.py` | `SpatialCalibration` class |
| `bemd.py` | `BEMD` class |
| `homomorphic_filter.py` | `HomomorphicFilter` class |
| `nonlinear_filtering.py` | `NonlinearFilter` class |
| `gamma_correction.py` | `ImageEnhancer.gamma_correction()` |
| `metrics.py` | `ImageMetrics` class |
| `image_resizer.py` | `ImageResizer` class |
| `main.ipynb` | `ImageProcessingPipeline.process()` |

## License

MIT License
