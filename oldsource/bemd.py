import cupy as cp
import gc
import logging
from cupyx.scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_local_extrema(image, window_size=3):
    """Get local maxima and minima maps for a given image."""
    # logging.info(f"Finding local extrema with window size: {window_size}")
    mask = image != 0
    max_map = (image == maximum_filter(image, size=window_size)) & mask
    min_map = (image == minimum_filter(image, size=window_size)) & mask
    return max_map, min_map


def apply_order_statistic_filter(image, extrema_map, filter_type="max", window_size=3):
    """Approximate the envelope using order statistics filters (MAX/MIN)"""
    # logging.info(f"Applying {filter_type} filter with window size: {window_size}")
    if filter_type == "max":
        envelope = maximum_filter(image, size=window_size)
    elif filter_type == "min":
        envelope = minimum_filter(image, size=window_size)
    else:
        raise ValueError("filter_type should be either 'max' or 'min'")
    return cp.where(extrema_map, envelope, image)


def smooth_envelope(envelope, smooth_window_size=3):
    """Smooth the envelope with an averaging filter."""
    # logging.info(f"Smoothing envelope with window size: {smooth_window_size}")
    return uniform_filter(envelope, size=smooth_window_size)


def calculate_mean_envelope(upper_envelope, lower_envelope):
    """Calculate the mean envelope."""
    # logging.info("Calculating mean envelope")
    return (upper_envelope + lower_envelope) / 2


def calculate_standard_deviation(FTj, FTj_next):
    """Calculate the standard deviation used for BIMF criteria checking."""
    # logging.info("Calculating standard deviation")
    return cp.sqrt(cp.sum((FTj_next - FTj) ** 2) / cp.sum(FTj**2))


def fabemd(image, max_iterations=10, threshold=0.2, initial_window_size=3, local_extrema_count=5):
    """
    Perform FABEMD decomposition on an input image.

    Args:
        image (np.ndarray): Input image (2D array).
        max_iterations (int): Maximum iterations for BIMF extraction.
        threshold (float): Threshold for standard deviation to accept BIMF.
        initial_window_size (int): Initial window size for finding extrema.

    Returns:
        list: A list of extracted BIMFs.
        np.ndarray: Residue of the decomposition.
    """
    #logging.info("Starting FABEMD decomposition")
    residual = image.astype(cp.float64)
    BIMFs = []
    lenBIMFs = len(BIMFs)
    limitSD = threshold

    # Iterate to extract each BIMF
    while True:
        FTj = residual.copy()
        window_size = initial_window_size
        SD = 1.0
        for j in range(max_iterations):
            #logging.info(f"Iteration {j+1}/{max_iterations}")

            # Step 1: Find local maxima and minima
            max_map, min_map = get_local_extrema(FTj, window_size=window_size)
            
            # Step 2: Estimate upper and lower envelopes using MAX/MIN filters
            upper_envelope = apply_order_statistic_filter(
                FTj, max_map, filter_type="max", window_size=window_size
            )
            lower_envelope = apply_order_statistic_filter(
                FTj, min_map, filter_type="min", window_size=window_size
            )
            
            # Step 3: Smooth the envelopes
            upper_envelope = smooth_envelope(
                upper_envelope, smooth_window_size=window_size
            )
            lower_envelope = smooth_envelope(
                lower_envelope, smooth_window_size=window_size
            )
            
            
            # Step 4: Calculate the mean envelope
            mean_envelope = calculate_mean_envelope(upper_envelope, lower_envelope)
            # print(mean_envelope)
            # print(lower_envelope)
            # Step 5: Update FTj for the next iteration
            FTj_next = FTj - mean_envelope
            SD = calculate_standard_deviation(FTj, FTj_next)

            # Check if the BIMF conditions are met
            if SD < limitSD:
                # logging.info(f"BIMF condition met with SD: {SD}")
                BIMFs.append(FTj_next)
                residual -= FTj_next
                break
            else:
                limitSD = 1.1 * limitSD
                FTj = FTj_next
                #window_size += 2  # Optionally adjust window size with each iteration

        # Stop if the residual has fewer than 3 extrema points
        max_map, min_map = get_local_extrema(residual)
        # logging.info(f"Residual has {cp.sum(max_map)} maxima and {cp.sum(min_map)} minima and {len(BIMFs)} BIMFs, SD = {SD}")
        print(f"Residual has {cp.sum(max_map)} maxima and {cp.sum(min_map)} minima and {len(BIMFs)} BIMFs, SD = {SD}", end="\r")
        if ((cp.sum(max_map) + cp.sum(min_map)) <= local_extrema_count):
            logging.info(f"Stopping criteria met: fewer than {local_extrema_count} extrema points")
            break
        elif (len(BIMFs) == 100):
            logging.info(f"Stopping criteria met: 100 BIMFs extracted")
            break
        
        del max_map, min_map
        gc.collect()
    logging.info("FABEMD decomposition completed")
    
    return BIMFs


# Usage:
# Assuming `input_image` is the (4096, 3255) grayscale image
# FABEMD decomposition on an example large image
# BIMFs, residue = fabemd(input_image)