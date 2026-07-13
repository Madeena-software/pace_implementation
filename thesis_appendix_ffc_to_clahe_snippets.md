# Appendix Code Snippets: FFC to CLAHE With FABEMD and MOO

The following appendix code is self-contained and does not import the existing `image_pipeline.py` or `fabemd.py` modules. It shows the actual implementation flow from Flat Field Correction (FFC) through CLAHE, including FABEMD decomposition and the MOO/grid-search score:

```python
total_score = CII + entropy + EME
```

## 1. Imports and GPU/CPU Compatibility

```python
import gc
import itertools
from pathlib import Path

import cv2
import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import (
        maximum_filter,
        median_filter,
        minimum_filter,
        uniform_filter,
        zoom,
    )
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    from scipy.ndimage import (
        maximum_filter,
        median_filter,
        minimum_filter,
        uniform_filter,
        zoom,
    )
    HAS_CUPY = False


def to_numpy(arr):
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return arr.get()
    return np.asarray(arr)
```

## 2. Flat Field Correction (FFC)

```python
def apply_ffc(projection_image, gain_image, dark_image, median_filter_size=7):
    proj = cp.asarray(projection_image)
    gain = cp.asarray(gain_image)
    dark = cp.asarray(dark_image)

    proj = (proj - cp.min(proj)) / (65535 - cp.min(proj))
    gain = (gain - cp.min(gain)) / (65535 - cp.min(gain))
    dark = (dark - cp.min(dark)) / (65535 - cp.min(dark))

    proj = median_filter(proj, median_filter_size)
    gain = median_filter(gain, median_filter_size)
    dark = median_filter(dark, median_filter_size)

    proj_dark = proj - dark
    gain_dark = gain - dark

    proj_dark[proj_dark <= 0] = 0
    gain_dark[gain_dark <= 0] = 1e-12

    intensity = proj_dark / gain_dark
    intensity[intensity <= 0] = 1e-12

    miu = -cp.log(intensity)
    miu = miu.astype(cp.float32)
    miu[miu < 0] = 0

    del proj, gain, dark, proj_dark, gain_dark, intensity
    gc.collect()

    return miu
```

## 3. Spatial Calibration

```python
def apply_spatial_calibration(image, calibration_path):
    with np.load(calibration_path) as params:
        mtx = params["mtx"]
        dist = params["dist"]
        roi = params["roi"]

    img = to_numpy(image)
    h, w = img.shape[:2]

    new_camera_mtx, roi_rotate = cv2.getOptimalNewCameraMatrix(
        mtx,
        dist,
        (w, h),
        1,
        (w, h),
    )

    undistorted = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    x, y, rw, rh = roi_rotate
    undistorted = undistorted[y:y + rh, x:x + rw]

    x, y, rw, rh = roi
    calibrated = undistorted[y:y + rh, x:x + rw]

    return calibrated
```

## 4. FABEMD Decomposition

```python
class FABEMD:
    def __init__(
        self,
        max_sift_iterations=1,
        sd_threshold=0.2,
        min_extrema=5,
        max_bimfs=20,
        initial_window_size=None,
        window_size_cap=2000,
        extrema_window=3,
        window_growth_rate=2.0,
    ):
        self.max_sift_iterations = max_sift_iterations
        self.sd_threshold = sd_threshold
        self.min_extrema = min_extrema
        self.max_bimfs = max_bimfs
        self.initial_window_size = initial_window_size
        self.window_size_cap = window_size_cap
        self.extrema_window = extrema_window
        self.window_growth_rate = window_growth_rate

    @staticmethod
    def normalize_window_size(window_size, cap):
        cap = max(3, int(cap))
        if cap % 2 == 0:
            cap -= 1

        window_size = int(window_size)
        window_size = max(3, min(window_size, cap))

        if window_size % 2 == 0:
            window_size += 1

        return min(window_size, cap)

    @staticmethod
    def find_local_extrema(image, window_size=3):
        mask = image != 0
        max_map = (image == maximum_filter(image, size=window_size)) & mask
        min_map = (image == minimum_filter(image, size=window_size)) & mask
        return max_map, min_map

    @staticmethod
    def compute_adaptive_window_size(extrema_map, image_shape, cap=2000):
        coords = cp.argwhere(extrema_map)
        count = int(coords.shape[0])

        if count < 2:
            return 3

        coords_np = to_numpy(coords)

        if len(coords_np) > 2000:
            sample_index = np.random.choice(len(coords_np), 2000, replace=False)
            coords_np = coords_np[sample_index]

        max_nearest_distance = 1.0

        for i, point in enumerate(coords_np):
            diff = coords_np - point
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            dist[i] = np.inf
            nearest = np.min(dist)
            if nearest > max_nearest_distance:
                max_nearest_distance = nearest

        window_size = int(np.ceil(2 * max_nearest_distance + 1))
        return FABEMD.normalize_window_size(window_size, cap)

    def adaptive_window(self, residual, previous_window=None):
        max_map, min_map = self.find_local_extrema(
            residual,
            window_size=self.extrema_window,
        )

        max_window = self.compute_adaptive_window_size(
            max_map,
            residual.shape,
            self.window_size_cap,
        )
        min_window = self.compute_adaptive_window_size(
            min_map,
            residual.shape,
            self.window_size_cap,
        )

        window_size = max(max_window, min_window)

        if self.initial_window_size is not None and previous_window is None:
            window_size = max(window_size, self.initial_window_size)

        if previous_window is not None:
            allowed = int(previous_window * self.window_growth_rate)
            window_size = min(window_size, allowed)

        return self.normalize_window_size(window_size, self.window_size_cap)

    @staticmethod
    def calculate_standard_deviation(old_image, new_image):
        numerator = cp.sum((new_image - old_image) ** 2)
        denominator = cp.sum(old_image ** 2)
        if float(denominator) == 0:
            return 0.0
        return float(cp.sqrt(numerator / denominator))

    def extract_bimf(self, residual, previous_window=None):
        current = residual.copy()
        window_size = self.adaptive_window(current, previous_window)

        for _ in range(self.max_sift_iterations):
            max_map, min_map = self.find_local_extrema(
                current,
                window_size=self.extrema_window,
            )

            upper_envelope = maximum_filter(current, size=window_size)
            lower_envelope = minimum_filter(current, size=window_size)

            upper_envelope = uniform_filter(upper_envelope, size=window_size)
            lower_envelope = uniform_filter(lower_envelope, size=window_size)

            mean_envelope = (upper_envelope + lower_envelope) / 2
            next_current = current - mean_envelope

            sd = self.calculate_standard_deviation(current, next_current)
            current = next_current

            if sd < self.sd_threshold:
                break

        return current, window_size

    def decompose_with_residual(self, image):
        residual = cp.asarray(image).astype(cp.float64)
        bimfs = []
        previous_window = None

        while len(bimfs) < self.max_bimfs:
            max_map, min_map = self.find_local_extrema(
                residual,
                window_size=self.extrema_window,
            )
            extrema_count = int(cp.sum(max_map) + cp.sum(min_map))

            if extrema_count < self.min_extrema:
                break

            bimf, previous_window = self.extract_bimf(
                residual,
                previous_window,
            )

            bimfs.append(bimf)
            residual = residual - bimf

            del max_map, min_map
            gc.collect()

        return bimfs, residual

    @staticmethod
    def calculate_energies(bimfs):
        energies = []
        for bimf in bimfs:
            energy = float(np.sum(np.square(to_numpy(bimf))))
            energies.append(energy)
        return energies
```

## 5. Homomorphic Filtering

```python
class HomomorphicFilter:
    def __init__(self, d0=40, rh=2.5, rl=0.99, c=1.0):
        self.d0 = d0
        self.rh = rh
        self.rl = rl
        self.c = c

    def apply(self, image, normalize=False):
        img = to_numpy(image).astype(np.float64)
        rows, cols = img.shape

        min_img = float(np.min(img))
        if min_img <= 0:
            img = img - min_img + 1e-10
        else:
            img = np.maximum(img, 1e-10)

        log_image = np.log(img)

        spectrum = np.fft.fft2(log_image)
        spectrum_shift = np.fft.fftshift(spectrum)

        u = np.arange(cols) - cols / 2
        v = np.arange(rows) - rows / 2
        U, V = np.meshgrid(u, v)
        distance = np.sqrt(U ** 2 + V ** 2)

        H = (
            (self.rh - self.rl)
            * (1 - np.exp(-self.c * (distance ** 2 / self.d0 ** 2)))
            + self.rl
        )

        filtered_spectrum = spectrum_shift * H
        filtered_log = np.real(
            np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))
        )

        result = np.exp(filtered_log)

        if normalize:
            rmin, rmax = result.min(), result.max()
            if rmax - rmin > 1e-10:
                result = (result - rmin) / (rmax - rmin)
            else:
                result = np.zeros_like(result)

        return result
```

## 6. Nonlinear Reconstruction

```python
def reconstruct_from_bimfs(
    bimfs,
    energies,
    filtered_residue,
    r=1,
    beta=1.0,
):
    sorted_indices = np.argsort(energies)

    denoised_bimfs = []
    for i in range(int(r)):
        index = sorted_indices[i]
        bimf_np = to_numpy(bimfs[index]).astype(np.float32)
        denoised = cv2.bilateralFilter(bimf_np, 5, 75, 75)
        denoised_bimfs.append(denoised)

    reconstructed_detail = np.sum(denoised_bimfs, axis=0)

    for j in range(int(r), len(bimfs)):
        index = sorted_indices[j]
        reconstructed_detail += to_numpy(bimfs[index])

    reconstructed = reconstructed_detail + beta * filtered_residue
    return reconstructed
```

## 7. Gamma Correction

```python
def gamma_correction(image, gamma=0.8):
    img = image.astype(np.float64)

    imin = img.min()
    imax = img.max()

    if imax - imin > 1e-10:
        img_normalized = (img - imin) / (imax - imin)
    else:
        img_normalized = np.zeros_like(img)

    img_corrected = np.power(np.clip(img_normalized, 0, 1), gamma)
    return np.uint16(img_corrected * 65535)
```

## 8. CLAHE

```python
def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(image)
```

## 9. Image Quality Metrics

```python
def calculate_contrast(image, mask):
    foreground = image[mask == 1]
    background = image[mask == 0]

    foreground_mean = np.mean(foreground) if len(foreground) > 0 else 0
    background_mean = np.mean(background) if len(background) > 0 else 0

    if foreground_mean + background_mean == 0:
        return 0.0

    return (
        (foreground_mean - background_mean)
        / (foreground_mean + background_mean)
    )


def calculate_cii(processed, reference, mask):
    processed_contrast = calculate_contrast(processed, mask)
    reference_contrast = calculate_contrast(reference, mask)

    if reference_contrast == 0:
        return 0.0

    return processed_contrast / reference_contrast


def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [65535], [0, 65535])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log(hist + 1e-7))
    return float(entropy)


def calculate_eme(image, row_blocks=4, col_blocks=4, epsilon=0.0001):
    height, width = image.shape
    block_height = height // row_blocks
    block_width = width // col_blocks

    eme = 0.0

    for i in range(row_blocks):
        for j in range(col_blocks):
            block = image[
                i * block_height:(i + 1) * block_height,
                j * block_width:(j + 1) * block_width,
            ]

            block_max = np.max(block)
            block_min = np.min(block)

            if block_min + epsilon == 0:
                continue

            contrast_ratio = block_max / (block_min + epsilon)
            eme += 20 * np.log(contrast_ratio)

    return eme / (row_blocks * col_blocks)
```

## 10. Single Parameter Evaluation

```python
def process_single_params(
    params,
    reference_image,
    bimfs,
    energies,
    residue,
    beta=1.0,
):
    d0, gH, rL, gamma, clip_limit, tile_grid_size = params

    homomorphic_filter = HomomorphicFilter(
        d0=d0,
        rh=gH,
        rl=rL,
    )

    filtered_residue = homomorphic_filter.apply(
        residue,
        normalize=False,
    )

    reconstructed = reconstruct_from_bimfs(
        bimfs,
        energies,
        filtered_residue,
        r=1,
        beta=beta,
    )

    gamma_img = gamma_correction(reconstructed, gamma)
    clahe_img = apply_clahe(
        gamma_img,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    mask = np.ones_like(reference_image)

    cii = calculate_cii(clahe_img, reference_image, mask)
    entropy = calculate_entropy(clahe_img)
    eme = calculate_eme(clahe_img, 4, 4)
    total_score = cii + entropy + eme

    return {
        "image": clahe_img,
        "params": params,
        "beta": beta,
        "cii": cii,
        "entropy": entropy,
        "eme": eme,
        "total_score": total_score,
    }
```

## 11. MOO / Grid Search

```python
def find_best_parameters(
    reference_image,
    bimfs,
    energies,
    residue,
    d0_values=(40,),
    gH_values=(2.5,),
    rL_values=(0.99,),
    beta_values=(1.0,),
    gamma_values=(0.8,),
    clip_limit_values=(3.0,),
    tile_grid_size_values=((8, 8),),
):
    parameter_combinations = list(itertools.product(
        d0_values,
        gH_values,
        rL_values,
        gamma_values,
        clip_limit_values,
        tile_grid_size_values,
    ))

    best_result = None
    all_results = []

    for beta in beta_values:
        for params in parameter_combinations:
            result = process_single_params(
                params,
                reference_image,
                bimfs,
                energies,
                residue,
                beta=beta,
            )
            all_results.append(result)

            if (
                best_result is None
                or result["total_score"] > best_result["total_score"]
            ):
                best_result = result

    all_results = sorted(
        all_results,
        key=lambda item: item["total_score"],
        reverse=True,
    )

    return best_result, all_results
```

## 12. Normalize, Resize, and Save

```python
def normalize_and_resize(image, output_width=4096):
    img = cp.asarray(image)

    min_val = cp.min(img)
    max_val = cp.max(img)

    img = min_val + ((img - min_val) / (max_val - min_val) * 65535)

    height, width = img.shape[:2]
    width_percent = output_width / float(width)
    output_height = int(height * width_percent)

    resized = zoom(
        img,
        (output_height / height, output_width / width),
        order=1,
    )

    return to_numpy(resized).astype(np.uint16)


def save_tiff(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_path),
        image.astype(np.uint16),
        params=(cv2.IMWRITE_TIFF_COMPRESSION, 1),
    )

    return output_path
```

## 13. Complete FFC to CLAHE Flow

```python
PROJ_PATH = "projection_image.tiff"
GAIN_PATH = "gain_image.mdn"
DARK_PATH = "dark_image.mdn"
CALIBRATION_PATH = "calibration_file.npz"
OUTPUT_PATH = "output/final_clahe_result.tiff"

proj_img = cv2.imread(str(PROJ_PATH), -1)
gain_img = cv2.imread(str(GAIN_PATH), -1)
dark_img = cv2.imread(str(DARK_PATH), -1)

if proj_img is None:
    raise FileNotFoundError(PROJ_PATH)
if gain_img is None:
    raise FileNotFoundError(GAIN_PATH)
if dark_img is None:
    raise FileNotFoundError(DARK_PATH)

ffc_img = apply_ffc(
    proj_img,
    gain_img,
    dark_img,
    median_filter_size=7,
)

calibrated_img = apply_spatial_calibration(
    ffc_img,
    CALIBRATION_PATH,
)

fabemd = FABEMD(
    max_sift_iterations=1,
    sd_threshold=0.2,
    min_extrema=5,
    max_bimfs=20,
    window_size_cap=2000,
    extrema_window=3,
    initial_window_size=None,
    window_growth_rate=2.0,
)

bimfs, residue = fabemd.decompose_with_residual(calibrated_img)
energies = fabemd.calculate_energies(bimfs)

best_result, all_results = find_best_parameters(
    calibrated_img,
    bimfs,
    energies,
    residue,
    d0_values=(40,),
    gH_values=(2.5,),
    rL_values=(0.99,),
    beta_values=(1.0,),
    gamma_values=(0.8,),
    clip_limit_values=(3.0,),
    tile_grid_size_values=((8, 8),),
)

final_image = normalize_and_resize(
    best_result["image"],
    output_width=4096,
)

save_tiff(final_image, OUTPUT_PATH)

print("Best parameters:", best_result["params"])
print("Beta:", best_result["beta"])
print("CII:", best_result["cii"])
print("Entropy:", best_result["entropy"])
print("EME:", best_result["eme"])
print("Total score:", best_result["total_score"])
```
