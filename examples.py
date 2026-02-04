"""
Example usage of the Image Processing Pipeline.

This script demonstrates how to use the refactored pipeline.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_pipeline import ImageProcessingPipeline, PipelineConfig


def example_basic_usage():
    """Basic usage with inline configuration."""
    
    # Create configuration
    config = PipelineConfig(
        # Image paths - update these to your actual paths
        proj_img_path="E:/DataBetaTest/07082024/36-KSW-36B/mentah/sample.mdn",
        gain_img_path="../datacitra/Gain/Bed/80_50_0,50.mdn",
        dark_img_path="../datacitra/Dark/Bed/dark.mdn",
        calibration_path="../datacitra/Kalibrasi/bed_44_35.npz",
        output_dir="../datacitra/output",
        
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
    
    # Create pipeline
    pipeline = ImageProcessingPipeline(config)
    
    # Run processing
    result = pipeline.process(show_plot=True)
    
    # Print results
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Best parameters: {result.parameters}")
    print(f"CII: {result.cii:.4f}")
    print(f"Entropy: {result.entropy:.4f}")
    print(f"EME: {result.eme:.4f}")
    print(f"Total Score: {result.total_score:.4f}")


def example_json_config():
    """Usage with JSON configuration file."""
    
    # Load configuration from JSON
    config = PipelineConfig.from_json("config.json")
    
    # Override specific values if needed
    config.num_threads = 4
    config.output_width = 2048
    
    # Create and run pipeline
    pipeline = ImageProcessingPipeline(config)
    result = pipeline.process(show_plot=False)
    
    return result


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
    
    # Configuration
    proj_path = "../datacitra/sample.tiff"
    gain_path = "../datacitra/Gain/Bed/80_50_0,50.mdn"
    dark_path = "../datacitra/Dark/Bed/dark.mdn"
    calib_path = "../datacitra/Kalibrasi/bed_44_35.npz"
    
    # Step 1: Load images
    print("Step 1: Loading images...")
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
        max_iterations=1,
        threshold=1.0,
        initial_window_size=32,
        local_extrema_count=10
    )
    bimfs = bemd.decompose(cp.asarray(calibrated_image))
    energies = BEMD.calculate_energies(bimfs)
    
    # Step 5: Apply Homomorphic Filter
    print("Step 5: Applying Homomorphic Filter...")
    hf = HomomorphicFilter(d0=30, rh=2.0, rl=0.5)
    filtered_image = hf.apply(calibrated_image)
    
    # Step 6: Denoise with Nonlinear Filter
    print("Step 6: Denoising...")
    nlf = NonlinearFilter(r=1, beta=0.5)
    denoised_image = nlf.denoise(bimfs, energies, filtered_image)
    
    # Step 7: Apply Gamma Correction
    print("Step 7: Applying Gamma Correction...")
    gamma_image = ImageEnhancer.gamma_correction(denoised_image, gamma=0.8)
    
    # Step 8: Apply CLAHE
    print("Step 8: Applying CLAHE...")
    clahe_image = ImageEnhancer.apply_clahe(gamma_image, clip_limit=3.0)
    
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
    final_image = ImageResizer.resize(clahe_image, 4096).get()
    cv2.imwrite("output_step_by_step.tiff", final_image.astype(np.uint16))
    
    print("\nProcessing complete!")
    return final_image


def example_batch_processing():
    """Process multiple images in batch."""
    
    import os
    from pathlib import Path
    
    # Configuration
    config = PipelineConfig.from_json("config.json")
    pipeline = ImageProcessingPipeline(config)
    
    # Input directory with projection images
    input_dir = "../datacitra/Thorax"
    output_dir = "../datacitra/output"
    
    # Process each image
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith(('.tiff', '.mdn', '.tif')):
            print(f"\nProcessing: {filename}")
            
            # Update paths
            proj_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, 
                Path(filename).stem + "_processed.tiff"
            )
            
            # Process
            result = pipeline.process(
                proj_path=proj_path,
                output_path=output_path,
                show_plot=False
            )
            
            results.append({
                'filename': filename,
                'score': result.total_score,
                'parameters': result.parameters
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"{r['filename']}: Score = {r['score']:.4f}")
    
    return results


if __name__ == "__main__":
    # Choose which example to run
    print("Image Processing Pipeline Examples")
    print("=" * 50)
    print("1. Basic usage with inline config")
    print("2. Usage with JSON config file")
    print("3. Step-by-step processing")
    print("4. Batch processing")
    print("=" * 50)
    
    choice = input("Select example (1-4): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_json_config()
    elif choice == "3":
        example_step_by_step()
    elif choice == "4":
        example_batch_processing()
    else:
        print("Running basic example by default...")
        example_basic_usage()
