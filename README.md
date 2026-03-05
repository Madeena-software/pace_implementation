# Image Processing Pipeline - Refactored

A unified, class-based image processing pipeline for medical imaging.

## Features

- **Modular Design**: Each processing step is encapsulated in its own class
- **Easy Configuration**: Use `PipelineConfig` dataclass or JSON files
- **GPU Acceleration**: Uses CuPy for GPU-accelerated processing (NumPy/SciPy fallback)
- **Parallel Processing**: ThreadPoolExecutor for parameter optimization
- **Type Hints**: Full type annotations for better IDE support
- **Logging**: Comprehensive logging throughout the pipeline
- **FABEMD**: Fast and Adaptive BEMD (Bhuiyan et al. 2008) as drop-in replacement for BEMD
- **PACE Homomorphic Filter**: Butterworth high-pass filter variant (Siracusano et al. 2020)
- **NL-Means Denoising**: Non-Local Means filter (PACE 2.0, Siracusano et al. 2023)

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy opencv-python scipy matplotlib
# Optional GPU acceleration:
pip install cupy-cuda12x
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
| `d0_values` | List[int] | [20, 30, 40] | Cutoff frequencies (Gaussian) |
| `rh_values` | List[float] | [1.5, 2.0, 2.5] | High freq gains (Gaussian) |
| `rl_values` | List[float] | [0.3, 0.5] | Low freq gains (Gaussian) |
| `gamma_values` | List[float] | [0.8] | Gamma values to try |
| `clip_limit_values` | List[float] | [3.0] | CLAHE clip limits to try |
| `tile_grid_size_values` | List[Tuple] | [(8, 8)] | CLAHE grid sizes to try |
| `denoise_r` | int | 1 | Number of BIMFs to denoise |
| `denoise_beta` | float | 0.5 | Residual weight in reconstruction |
| `denoise_method` | str | "bilateral" | "bilateral" or "nlmeans" |
| `nlmeans_h` | float | 10.0 | NL-means filter strength |
| `nlmeans_template_window` | int | 7 | NL-means patch size |
| `nlmeans_search_window` | int | 21 | NL-means search area size |
| `decomposition_method` | str | "fabemd" | "bemd" or "fabemd" |
| `fabemd_max_sift_iterations` | int | 10 | FABEMD max sifting iterations |
| `fabemd_sd_threshold` | float | 0.2 | FABEMD SD stopping threshold |
| `fabemd_min_extrema` | int | 5 | FABEMD min extrema to stop |
| `fabemd_max_bimfs` | int | 100 | FABEMD max BIMFs to extract |
| `homomorphic_method` | str | "gaussian" | "gaussian" or "butterworth" |
| `pace_d0_values` | List[int] | [20, 30, 40] | Cutoff frequencies (Butterworth) |
| `pace_gamma_h_values` | List[float] | [1.5, 2.0, 2.5] | High freq gains (Butterworth) |
| `pace_gamma_l_values` | List[float] | [0.3, 0.5] | Low freq gains (Butterworth) |
| `pace_n_values` | List[int] | [1, 2] | Butterworth filter orders |
| `output_width` | int | 4096 | Target output width |
| `num_threads` | int | 8 | Number of parallel threads |
| `processing_mode` | str | "full" | "full" or "pace" |

## Processing Pipeline

1. **Load Images**: Load projection, gain, and dark images
2. **Flat Field Correction**: Apply FFC using GPU-accelerated median filtering
3. **Spatial Calibration**: Apply lens distortion correction
4. **Decomposition**: Extract BIMFs via BEMD or FABEMD (`decomposition_method`)
5. **Parameter Optimization**: Find best processing parameters through grid search
   - Homomorphic filtering — Gaussian (`HomomorphicFilter`) or Butterworth (`PACEHomomorphicFilter`)
   - Nonlinear filtering — Bilateral (`NonlinearFilter`) or NL-Means (`NLMeansFilter`)
   - Gamma correction
   - CLAHE enhancement
6. **Evaluation**: Calculate CII, entropy, and EME metrics
7. **Output**: Normalize, resize, and save the result

### PACE Processing Flow

1. **Load Image**: Load projection image only
2. **Decomposition**: Extract BIMFs via BEMD or FABEMD
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
    PACEHomomorphicFilter,
    NonlinearFilter,
    NLMeansFilter,
    ImageEnhancer,
    ImageMetrics,
)
from fabemd import FABEMD

# Flat Field Correction
ffc = FlatFieldCorrection(median_filter_size=7)
corrected = ffc.apply(proj_img, gain_img, dark_img)

# BEMD Decomposition
bemd = BEMD(max_iterations=1, threshold=1.0)
bimfs = bemd.decompose(cp.asarray(image))
energies = BEMD.calculate_energies(bimfs)

# FABEMD Decomposition (alternative)
fabemd = FABEMD(max_sift_iterations=10, sd_threshold=0.2)
bimfs = fabemd.decompose(cp.asarray(image))
energies = FABEMD.calculate_energies(bimfs)

# Gaussian Homomorphic Filter (classic)
hf = HomomorphicFilter(d0=30, rh=2.0, rl=0.5)
filtered = hf.apply(image)

# Butterworth Homomorphic Filter (PACE)
pace_hf = PACEHomomorphicFilter(d0=30, gamma_h=2.0, gamma_l=0.5, n=2)
filtered = pace_hf.apply(image)

# Bilateral denoising (original)
nlf = NonlinearFilter(r=1, beta=0.5)
denoised = nlf.denoise(bimfs, energies, filtered)

# NL-Means denoising (PACE 2.0)
nlm = NLMeansFilter(r=1, beta=0.5, h=10.0)
denoised = nlm.denoise(bimfs, energies, filtered)

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
pace_implementation/
├── __init__.py          # Package initialization with exports
├── image_pipeline.py    # Main module with all classes
├── fabemd.py            # FABEMD decomposition engine
├── config.json          # Example configuration file
├── colab-setup.ipynb    # Interactive Colab-ready examples
├── playground.ipynb     # Step-by-step pipeline walkthrough
├── fabemd.ipynb         # FABEMD decomposition demo
├── examples.py          # Script-based examples
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
| `bemd.py` | `BEMD` class + `FABEMD` (fabemd.py) |
| `homomorphic_filter.py` | `HomomorphicFilter` + `PACEHomomorphicFilter` |
| `nonlinear_filtering.py` | `NonlinearFilter` + `NLMeansFilter` |
| `gamma_correction.py` | `ImageEnhancer.gamma_correction()` |
| `metrics.py` | `ImageMetrics` class |
| `image_resizer.py` | `ImageResizer` class |
| `main.ipynb` | `ImageProcessingPipeline.process()` |

## References

- **PACE**: Siracusano, G. et al. "Pipeline for Advanced Contrast Enhancement (PACE) of Chest X-ray in Evaluating COVID-19 Patients." *J. Digit. Imaging*, 2020.
- **PACE 2.0**: Siracusano, G. et al. "Effective processing pipeline PACE 2.0 for enhancing chest x-ray contrast and diagnostic interpretability." *Scientific Reports*, 2023.
- **FABEMD**: Bhuiyan, S.M.A., Adhami, R.R., Khan, J.F. "A novel approach of fast and adaptive bidimensional empirical mode decomposition." *IEEE ICASSP*, 2008.

## License

MIT License
