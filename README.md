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

### Image Paths

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `proj_img_path` | str | "" | Path to the raw projection (X-ray) image file (.mdn or .tiff). This is the primary input image to be processed. |
| `gain_img_path` | str | "" | Path to the gain (flat-field) calibration image. Captured with uniform illumination to record per-pixel detector sensitivity. Used in FFC to normalize brightness. |
| `dark_img_path` | str | "" | Path to the dark-current calibration image. Captured with no illumination to record baseline electronic noise. Subtracted from both projection and gain during FFC. |
| `calibration_path` | str | "" | Path to the spatial calibration file (.npz). Contains `camera_matrix`, `dist_coeffs`, and `roi` generated from checkerboard calibration. Used to undistort and crop the image. |
| `output_dir` | str | "" | Directory where processed output images will be saved. Created automatically if it doesn't exist. |

### Processing Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `processing_mode` | str | "full" | Controls the pipeline workflow. **"full"**: applies FFC + spatial calibration before decomposition (requires gain, dark, and calibration files). **"pace"**: skips FFC and calibration, starts directly from decomposition (useful when the input image is already corrected). |

### Flat Field Correction (FFC)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ffc_median_filter_size` | int | 7 | Size of the median filter kernel applied to the gain image before FFC. Removes salt-and-pepper noise from the gain image. Must be an odd number. Larger values = more smoothing of detector artifacts but may blur fine details. Typical range: 3–11. |

### Decomposition

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decomposition_method` | str | "fabemd" | Selects the decomposition algorithm. **"bemd"**: classic Bidimensional EMD with iterative surface interpolation — slower but well-studied. **"fabemd"**: Fast and Adaptive BEMD (Bhuiyan et al. 2008) using order-statistics filters with adaptive windows — significantly faster, recommended for large images. |

#### BEMD Parameters (used when `decomposition_method = "bemd"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bemd_max_iterations` | int | 1 | Maximum number of sifting iterations per BIMF extraction. Each iteration refines the BIMF by subtracting the mean envelope. Higher values yield more refined BIMFs but increase computation time. For medical images, 1–3 is usually sufficient. |
| `bemd_threshold` | float | 1.0 | Stopping criterion for the sifting process. The sifting stops when the standard deviation between successive iterations falls below this threshold. Lower values = stricter convergence = more refined BIMFs. Range: 0.1–2.0. |
| `bemd_initial_window_size` | int | 32 | Initial window size (in pixels) for local extrema detection. Determines the scale of features captured in the first BIMF. Smaller windows capture finer details first; larger windows start with coarser structures. Typical range: 16–64. |
| `bemd_local_extrema_count` | int | 10 | Minimum number of local extrema required to continue decomposition. When the residue has fewer extrema than this threshold, decomposition stops. Higher values stop earlier (fewer BIMFs); lower values extract more components. Typical range: 5–20. |

#### FABEMD Parameters (used when `decomposition_method = "fabemd"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fabemd_max_sift_iterations` | int | 10 | Maximum number of sifting iterations per BIMF. Each iteration applies order-statistics filters to estimate upper/lower envelopes. More iterations yield a more refined BIMF but take longer. Typical range: 1–20. |
| `fabemd_sd_threshold` | float | 0.2 | Standard deviation stopping criterion for sifting. The sifting loop for a BIMF stops when consecutive iterations differ by less than this value (relative energy). Lower values = more aggressive refinement. Typical range: 0.05–0.5. |
| `fabemd_min_extrema` | int | 5 | Minimum number of local extrema in the residue to continue extracting BIMFs. Below this count the image is considered a monotonic trend (residue) and decomposition stops. Lower values extract more BIMFs. Typical range: 2–10. |
| `fabemd_max_bimfs` | int | 100 | Hard cap on the number of BIMFs to extract. Prevents runaway decomposition. The actual count is usually much lower as `fabemd_min_extrema` stops the process first. Set high (50–200) to let the natural stopping control apply. |
| `fabemd_window_size_cap` | int | 201 | Maximum allowable window size (in pixels) for the adaptive order-statistics filters. Caps the envelope smoothing window to prevent it from growing too large on big images. Must be odd. Typical range: 101–501. |
| `fabemd_extrema_window` | int | 3 | Window size for local extrema detection via `maximum_filter` / `minimum_filter`. Controls how far apart detected peaks/valleys must be. Larger values skip smaller fluctuations. Must be odd. Typical range: 3–9. |
| `fabemd_initial_window_size` | int | None | If set, overrides the automatic initial window calculation. By default FABEMD computes the initial window from extrema spacing. Set only if you want manual control. |

### Homomorphic Filtering

Homomorphic filtering operates in the log-frequency domain. The image is log-transformed, Fourier-transformed, filtered with an emphasis function, inverse-transformed, and exponentiated. This separates illumination (low-frequency) from reflectance (high-frequency), allowing you to compress illumination variation while boosting anatomical detail.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `homomorphic_method` | str | "gaussian" | Selects the high-pass emphasis filter shape. **"gaussian"**: smooth roll-off, classic implementation. **"butterworth"**: sharper roll-off with adjustable order, as described in PACE (Siracusano et al. 2020). Butterworth is recommended for chest X-rays. |

#### Gaussian Homomorphic Parameters (used when `homomorphic_method = "gaussian"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d0_values` | List[int] | [20, 30, 40] | Cutoff frequency D₀ in pixels (distance from the center of the frequency domain). Controls the transition between suppressed low frequencies (illumination) and passed high frequencies (detail). Lower D₀ = more aggressive illumination flattening but may remove useful gradients. Higher D₀ = preserves more low-frequency structure. Typical range: 10–80. |
| `rh_values` | List[float] | [1.5, 2.0, 2.5] | High-frequency gain (γ_H). Multiplier applied to frequencies above D₀. Values > 1.0 amplify fine details and edges. Too high = noise amplification. Typical range: 1.2–3.0. |
| `rl_values` | List[float] | [0.3, 0.5] | Low-frequency gain (γ_L). Multiplier applied to frequencies below D₀. Values < 1.0 suppress slow illumination variations, flattening the background. 0 = full suppression; 1 = no change. Typical range: 0.1–0.9. |

#### Butterworth (PACE) Homomorphic Parameters (used when `homomorphic_method = "butterworth"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pace_d0_values` | List[int] | [20, 30, 40] | Same meaning as `d0_values` but used with the Butterworth transfer function. Controls the cutoff frequency for the Butterworth high-pass filter. |
| `pace_gamma_h_values` | List[float] | [1.5, 2.0, 2.5] | High-frequency gain (γ_H) for the Butterworth filter. Amplifies detail above the cutoff. Typical range: 1.2–3.0. |
| `pace_gamma_l_values` | List[float] | [0.3, 0.5] | Low-frequency gain (γ_L) for the Butterworth filter. Suppresses illumination below the cutoff. Values close to 1.0 (e.g. 0.99) preserve more of the original lighting. Typical range: 0.1–1.0. |
| `pace_n_values` | List[int] | [1, 2] | Butterworth filter order (n). Controls the sharpness of the transition from γ_L to γ_H around D₀. Order 1 = gradual roll-off (similar to Gaussian); order 2+ = steep, well-defined cutoff. Higher order = sharper separation but may introduce ringing. Typical range: 1–4. |
| `butterworth_order` | int | 2 | Default Butterworth order used when a single value is needed (e.g. in the playground notebook). Same effect as `pace_n_values`. |

### Enhancement Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma_values` | List[float] | [0.8] | Gamma correction exponent. Applied as: `output = input^γ`. **γ < 1**: brightens the image (expands dark tones, compresses bright tones) — useful for dark medical images. **γ > 1**: darkens the image. **γ = 1**: no change. Typical range for medical X-rays: 0.5–1.0. |
| `clip_limit_values` | List[float] | [3.0] | CLAHE contrast clip limit. Controls the maximum contrast amplification in each tile. Higher values allow more contrast enhancement but may amplify noise. Lower values produce more subtle enhancement. 1.0 = no enhancement; 40.0 = maximum. Typical range: 2.0–5.0. |
| `tile_grid_size_values` | List[Tuple] | [(8, 8)] | CLAHE tile grid size as (rows, cols). The image is divided into this many tiles, each equalized independently. Smaller tiles (more divisions) produce more local contrast but may introduce tile-boundary artifacts. Larger tiles approach global histogram equalization. Typical: (4,4) to (16,16). |

### Nonlinear Filtering (Denoising & Reconstruction)

After decomposition, the R lowest-energy BIMFs are denoised and then all BIMFs are recombined with the homomorphic-filtered residue:

$$I_{reconstructed} = \sum_{i} \text{BIMF}_i^{*} + \beta \times \text{filtered\_residue}$$

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `denoise_method` | str | "bilateral" | Selects the denoising back-end. **"bilateral"**: edge-preserving bilateral filter — fast, good at smoothing flat regions while keeping edges. **"nlmeans"**: Non-Local Means (PACE 2.0, Siracusano et al. 2023) — searches for similar patches across the image, better texture preservation but slower. |
| `denoise_r` | int | 1 | Number of lowest-energy BIMFs to denoise (R). The BIMFs are sorted by energy; the R components with least energy (typically high-frequency noise) are filtered. The remaining BIMFs pass through unchanged. Higher R = more denoising but may remove fine detail. Typical range: 1–3. |
| `denoise_beta` | float | 0.5 | Weight (β) of the homomorphic-filtered residue in the reconstruction. Controls how strongly the filtered residue contributes to the final image. 0 = residue omitted; 1 = full contribution. Values > 1 amplify the residue. Typical range: 0.3–1.5. |

#### NL-Means Parameters (used when `denoise_method = "nlmeans"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nlmeans_h` | float | 10.0 | Filter strength (h). Controls the degree of denoising. Higher values remove more noise but may blur details. The filtering weight between two patches decays as exp(-‖p₁−p₂‖² / h²), so h acts as the "noise tolerance". Typical range: 5–30 (depends on noise level). |
| `nlmeans_template_window` | int | 7 | Patch size for comparison (template window). Size in pixels of the patches compared between pixels. Larger patches capture more context for similarity but are slower and may over-smooth fine textures. Must be odd. Typical range: 5–11. |
| `nlmeans_search_window` | int | 21 | Search area size. Size in pixels of the neighborhood searched for similar patches. Larger search areas find better matches (better denoising) but are significantly slower (O(search²) per pixel). Must be odd. Typical range: 11–35. |

### Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_width` | int | 4096 | Target width (in pixels) for the final output image. The image is resized to this width while maintaining aspect ratio using GPU-accelerated bilinear interpolation. Set to 0 to skip resizing. |
| `num_threads` | int | 8 | Number of parallel threads for grid-search parameter optimization. More threads = faster search but higher memory usage. Set to 1 for debugging. Typical: 4–16 depending on available CPU cores. |

## Processing Pipeline

The full pipeline processes a raw detector image through these stages:

1. **Load Images**: Load projection, gain, and dark images from disk (.mdn or .tiff format)
2. **Flat Field Correction (FFC)**: Normalizes detector response using $\mu = -\ln\left(\frac{I_{proj} - I_{dark}}{I_{gain} - I_{dark}}\right)$. The gain image is median-filtered first (`ffc_median_filter_size`) to remove hot/dead pixels.
3. **Spatial Calibration**: Corrects lens distortion using pre-computed camera matrix and distortion coefficients from the calibration file. The image is undistorted and cropped to the valid ROI.
4. **Decomposition**: Decomposes the calibrated image into BIMFs (frequency layers) and a residue using BEMD or FABEMD (`decomposition_method`). BIMFs capture detail at different scales; the residue holds the smooth background.
5. **Homomorphic Filtering on Residue**: The decomposition residue is filtered in the log-frequency domain to flatten illumination and enhance anatomical detail. Choose Gaussian or Butterworth (`homomorphic_method`).
6. **Nonlinear Denoising & Reconstruction**: The R lowest-energy BIMFs are denoised (bilateral or NL-means), then all BIMFs are recombined with the filtered residue using weight β.
7. **Gamma Correction**: Adjusts overall brightness ($\gamma < 1$ brightens).
8. **CLAHE Enhancement**: Adaptive local contrast enhancement with clip limiting to prevent noise amplification.
9. **Evaluation**: Calculates image quality metrics — CII, entropy, and EME.
10. **Output**: Normalizes to 16-bit, resizes to `output_width`, and saves.

### PACE Processing Flow

Skips FFC and spatial calibration (set `processing_mode = "pace"`):

1. **Load Image**: Load projection image only (already corrected)
2. **Decomposition**: Extract BIMFs via BEMD or FABEMD
3. **Homomorphic Filtering on Residue**: Flatten illumination on the residue
4. **Denoising & Reconstruction**: Denoise low-energy BIMFs + reconstruct with filtered residue
5. **Enhancement**: Gamma correction + CLAHE
6. **Evaluation**: CII, entropy, and EME metrics
7. **Output**: Normalize, resize, and save

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
