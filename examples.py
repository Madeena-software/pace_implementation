"""
Example usage of the Image Processing Pipeline.

This script demonstrates how to use the refactored pipeline.
Configuration settings are gathered interactively from the user.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_pipeline import ImageProcessingPipeline, PipelineConfig


# =============================================================================
# Input helpers
# =============================================================================

def _input_str(prompt: str, default: str = "") -> str:
    """Prompt user for a string value with a default."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def _input_int(prompt: str, default: int = 0) -> int:
    """Prompt user for an integer value with a default."""
    value = input(f"{prompt} [{default}]: ").strip()
    return int(value) if value else default


def _input_float(prompt: str, default: float = 0.0) -> float:
    """Prompt user for a float value with a default."""
    value = input(f"{prompt} [{default}]: ").strip()
    return float(value) if value else default


def _input_int_list(prompt: str, default: list) -> list:
    """Prompt user for a comma-separated list of ints."""
    default_str = ", ".join(str(v) for v in default)
    value = input(f"{prompt} [{default_str}]: ").strip()
    if not value:
        return default
    return [int(x.strip()) for x in value.split(",")]


def _input_float_list(prompt: str, default: list) -> list:
    """Prompt user for a comma-separated list of floats."""
    default_str = ", ".join(str(v) for v in default)
    value = input(f"{prompt} [{default_str}]: ").strip()
    if not value:
        return default
    return [float(x.strip()) for x in value.split(",")]


def _input_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt user for a yes/no answer."""
    suffix = " [Y/n]" if default else " [y/N]"
    value = input(f"{prompt}{suffix}: ").strip().lower()
    if not value:
        return default
    return value in ("y", "yes")


# =============================================================================
# Interactive configuration builder
# =============================================================================

def _get_common_config() -> dict:
    """Gather BEMD, enhancement, and output parameters from the user."""
    print("\n--- BEMD Parameters ---")
    bemd_max_iterations = _input_int("  Max iterations", 1)
    bemd_threshold = _input_float("  Threshold", 1.0)
    bemd_initial_window_size = _input_int("  Initial window size", 32)
    bemd_local_extrema_count = _input_int("  Local extrema count", 10)

    print("\n--- Parameter Search Ranges ---")
    d0_values = _input_int_list("  D0 values (comma-separated)", [20, 30, 40])
    rh_values = _input_float_list("  Rh values (comma-separated)", [1.5, 2.0, 2.5])
    rl_values = _input_float_list("  Rl values (comma-separated)", [0.3, 0.5])
    gamma_values = _input_float_list("  Gamma values (comma-separated)", [0.8])
    clip_limit_values = _input_float_list("  CLAHE clip limit values (comma-separated)", [3.0])

    print("\n--- Output Settings ---")
    output_width = _input_int("  Output width (px)", 4096)
    num_threads = _input_int("  Number of threads", 8)
    output_dir = _input_str("  Output directory", "../datacitra/output")

    return dict(
        bemd_max_iterations=bemd_max_iterations,
        bemd_threshold=bemd_threshold,
        bemd_initial_window_size=bemd_initial_window_size,
        bemd_local_extrema_count=bemd_local_extrema_count,
        d0_values=d0_values,
        rh_values=rh_values,
        rl_values=rl_values,
        gamma_values=gamma_values,
        clip_limit_values=clip_limit_values,
        tile_grid_size_values=[(8, 8)],
        output_width=output_width,
        num_threads=num_threads,
        output_dir=output_dir,
    )


def _get_full_paths() -> dict:
    """Gather all image paths needed for the full pipeline (FFC + calibration)."""
    print("\n--- Image Paths ---")
    proj_img_path = _input_str("  Projection image path")
    gain_img_path = _input_str("  Gain image path", "../datacitra/Gain/Bed/80_50_0,50.mdn")
    dark_img_path = _input_str("  Dark image path", "../datacitra/Dark/Bed/dark.mdn")
    calibration_path = _input_str("  Calibration file path (.npz)", "../datacitra/Kalibrasi/bed_44_35.npz")
    return dict(
        proj_img_path=proj_img_path,
        gain_img_path=gain_img_path,
        dark_img_path=dark_img_path,
        calibration_path=calibration_path,
    )


def _get_pace_paths() -> dict:
    """Gather the projection image path only (PACE mode)."""
    print("\n--- Image Path (PACE mode — no FFC / calibration) ---")
    proj_img_path = _input_str("  Projection image path")
    return dict(proj_img_path=proj_img_path)


def _print_results(result) -> None:
    """Print processing results."""
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Best parameters: {result.parameters}")
    print(f"CII:         {result.cii:.4f}")
    print(f"Entropy:     {result.entropy:.4f}")
    print(f"EME:         {result.eme:.4f}")
    print(f"Total Score: {result.total_score:.4f}")


# =============================================================================
# Example functions
# =============================================================================

def example_basic_usage():
    """Full pipeline with user-supplied configuration."""
    paths = _get_full_paths()
    common = _get_common_config()

    config = PipelineConfig(**paths, **common, processing_mode="full")
    pipeline = ImageProcessingPipeline(config)
    result = pipeline.process(show_plot=True)
    _print_results(result)


def example_json_config():
    """Load base config from JSON, let user override key values."""
    json_path = _input_str("JSON config file path", "config.json")
    config = PipelineConfig.from_json(json_path)

    print("\nOverride settings (press Enter to keep current value):")
    config.num_threads = _input_int("  Number of threads", config.num_threads)
    config.output_width = _input_int("  Output width (px)", config.output_width)

    pipeline = ImageProcessingPipeline(config)
    result = pipeline.process(show_plot=False)
    _print_results(result)
    return result


def example_pace_usage():
    """PACE mode: skip FFC and spatial calibration."""
    paths = _get_pace_paths()
    common = _get_common_config()

    config = PipelineConfig(**paths, **common, processing_mode="pace")
    pipeline = ImageProcessingPipeline(config)
    result = pipeline.process(show_plot=True, mode="pace")
    _print_results(result)


def example_step_by_step():
    """Step-by-step processing for more control."""

    from image_pipeline import (
        FlatFieldCorrection,
        SpatialCalibration,
        BEMD,
        HomomorphicFilter,
        NonlinearFilter,
        ImageEnhancer,
        ImageMetrics,
        ImageResizer,
    )
    import cv2
    import cupy as cp
    import numpy as np

    print("\n--- Image Paths ---")
    proj_path = _input_str("  Projection image path", "../datacitra/sample.tiff")
    gain_path = _input_str("  Gain image path", "../datacitra/Gain/Bed/80_50_0,50.mdn")
    dark_path = _input_str("  Dark image path", "../datacitra/Dark/Bed/dark.mdn")
    calib_path = _input_str("  Calibration file path (.npz)", "../datacitra/Kalibrasi/bed_44_35.npz")

    print("\n--- BEMD Parameters ---")
    max_iter = _input_int("  Max iterations", 1)
    threshold = _input_float("  Threshold", 1.0)
    window_size = _input_int("  Initial window size", 32)
    extrema_count = _input_int("  Local extrema count", 10)

    print("\n--- Filter Parameters ---")
    d0 = _input_int("  Homomorphic D0", 30)
    rh = _input_float("  Homomorphic Rh", 2.0)
    rl = _input_float("  Homomorphic Rl", 0.5)
    gamma = _input_float("  Gamma correction", 0.8)
    clip_limit = _input_float("  CLAHE clip limit", 3.0)
    out_width = _input_int("  Output width (px)", 4096)

    # Step 1: Load images
    print("\nStep 1: Loading images...")
    proj_img = cv2.imread(proj_path, -1)
    gain_img = cv2.imread(gain_path, -1)
    dark_img = cv2.imread(dark_path, -1)

    # Step 2: Apply Flat Field Correction
    print("Step 2: Applying Flat Field Correction...")
    ffc = FlatFieldCorrection(median_filter_size=7)
    ffc_image = ffc.apply(proj_img, gain_img, dark_img)

    # Step 3: Apply Spatial Calibration
    print("Step 3: Applying Spatial Calibration...")
    calibrator = SpatialCalibration(calib_path)
    calibrated_image = calibrator.apply(ffc_image)

    # Step 4: BEMD Decomposition
    print("Step 4: BEMD Decomposition...")
    bemd = BEMD(
        max_iterations=max_iter,
        threshold=threshold,
        initial_window_size=window_size,
        local_extrema_count=extrema_count,
    )
    bimfs = bemd.decompose(cp.asarray(calibrated_image))
    energies = BEMD.calculate_energies(bimfs)

    # Step 5: Apply Homomorphic Filter
    print("Step 5: Applying Homomorphic Filter...")
    hf = HomomorphicFilter(d0=d0, rh=rh, rl=rl)
    filtered_image = hf.apply(calibrated_image)

    # Step 6: Denoise with Nonlinear Filter
    print("Step 6: Denoising...")
    nlf = NonlinearFilter(r=1, beta=0.5)
    denoised_image = nlf.denoise(bimfs, energies, filtered_image)

    # Step 7: Apply Gamma Correction
    print("Step 7: Applying Gamma Correction...")
    gamma_image = ImageEnhancer.gamma_correction(denoised_image, gamma=gamma)

    # Step 8: Apply CLAHE
    print("Step 8: Applying CLAHE...")
    clahe_image = ImageEnhancer.apply_clahe(gamma_image, clip_limit=clip_limit)

    # Step 9: Calculate Metrics
    print("Step 9: Calculating Metrics...")
    mask = np.ones_like(calibrated_image)
    cii = ImageMetrics.calculate_cii(clahe_image, calibrated_image, mask)
    entropy = ImageMetrics.calculate_entropy(clahe_image)
    eme = ImageMetrics.calculate_eme(clahe_image, 4, 4)

    print(f"\nMetrics:")
    print(f"  CII: {cii:.4f}")
    print(f"  Entropy: {entropy:.4f}")
    print(f"  EME: {eme:.4f}")

    # Step 10: Resize and Save
    print("Step 10: Resizing and saving...")
    final_image = ImageResizer.resize(clahe_image, out_width).get()
    cv2.imwrite("output_step_by_step.tiff", final_image.astype(np.uint16))

    print("\nProcessing complete!")
    return final_image


def example_batch_processing():
    """Process multiple images in batch (full pipeline)."""
    from pathlib import Path

    print("\n--- Batch Configuration (Full Pipeline) ---")
    input_dir = _input_str("  Input directory", "../datacitra/Thorax")
    output_dir = _input_str("  Output directory", "../datacitra/output")

    use_json = _input_yes_no("  Load base config from JSON?", True)
    if use_json:
        json_path = _input_str("  JSON config file path", "config.json")
        config = PipelineConfig.from_json(json_path)
    else:
        paths = _get_full_paths()
        common = _get_common_config()
        config = PipelineConfig(**paths, **common, processing_mode="full")

    config.output_dir = output_dir
    pipeline = ImageProcessingPipeline(config)

    results = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.tiff', '.mdn', '.tif')):
            print(f"\nProcessing: {filename}")
            proj_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir,
                Path(filename).stem + "_processed.tiff"
            )
            result = pipeline.process(
                proj_path=proj_path,
                output_path=output_path,
                show_plot=False,
            )
            results.append({
                'filename': filename,
                'score': result.total_score,
                'parameters': result.parameters,
            })

    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"{r['filename']}: Score = {r['score']:.4f}")
    return results


def example_pace_batch_processing():
    """Batch processing in PACE mode (no FFC or spatial calibration)."""
    from pathlib import Path

    print("\n--- Batch Configuration (PACE Mode) ---")
    input_dir = _input_str("  Input directory", "../datacitra/Thorax")
    output_dir = _input_str("  Output directory", "../datacitra/output")

    use_json = _input_yes_no("  Load base config from JSON?", True)
    if use_json:
        json_path = _input_str("  JSON config file path", "config.json")
        config = PipelineConfig.from_json(json_path)
    else:
        common = _get_common_config()
        config = PipelineConfig(**common)

    config.processing_mode = "pace"
    config.output_dir = output_dir
    pipeline = ImageProcessingPipeline(config)

    results = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".tiff", ".mdn", ".tif")):
            print(f"\nProcessing (PACE): {filename}")
            proj_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir,
                Path(filename).stem + "_pace_processed.tiff"
            )
            result = pipeline.process(
                proj_path=proj_path,
                output_path=output_path,
                show_plot=False,
                mode="pace",
            )
            results.append({
                "filename": filename,
                "score": result.total_score,
                "parameters": result.parameters,
            })

    print("\n" + "=" * 50)
    print("PACE BATCH PROCESSING SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"{r['filename']}: Score = {r['score']:.4f}")
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Image Processing Pipeline Examples")
    print("=" * 50)
    print("1. Full pipeline (interactive config)")
    print("2. Load from JSON config")
    print("3. Step-by-step processing")
    print("4. Batch processing (full)")
    print("5. PACE mode (no FFC + calibration)")
    print("6. PACE batch processing")
    print("=" * 50)

    choice = input("Select example (1-6): ").strip()

    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_json_config()
    elif choice == "3":
        example_step_by_step()
    elif choice == "4":
        example_batch_processing()
    elif choice == "5":
        example_pace_usage()
    elif choice == "6":
        example_pace_batch_processing()
    else:
        print("Running basic example by default...")
        example_basic_usage()
