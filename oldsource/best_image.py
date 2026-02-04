import cv2
import itertools
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

import homomorphic_filter as hf
import nonlinear_filtering as nf



# Define the functions (assuming they are defined elsewhere in the notebook)
# def homomorphic_filter(image, d0=30, rh=2.0, rl=0.5, c=1.0):
# def calculate_bimf_energy(bimfs):
# def reconstruct_image(bimfs, energies, R, beta, filtered_residual):
# def gamma_correction(image, gamma):
# def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
# def calculate_cii(image, reference_image, mask):
# def calculate_entropy(image):
# def calculate_eme(image, r, c):

# Define parameter ranges
d0_values = [20, 30, 40]
rh_values = [1.5, 2.0, 2.5]
rl_values = [0.3, 0.5]
gamma_values = [0.8]
clip_limit_values = [3.0]
tile_grid_size_values = [(8, 8)]

# Generate parameter combinations
parameter_combinations = list(
    itertools.product(
        d0_values,
        rh_values,
        rl_values,
        gamma_values,
        clip_limit_values,
        tile_grid_size_values,
    )
)

print("Total parameter combinations:", len(parameter_combinations))

# Initialize variables to store evaluation results
best_cii = 0.0
best_entropy = 0.0
best_emr = 0.0
best_parameters = None
best_image = None

# Load the reference image
# reference_image = cv2.imread("image_sample/1-IMA-01B_Thorax_AP.tiff", cv2.IMREAD_GRAYSCALE)
reference_image = image

def process_parameters(params):
    d0, rh, rl, gamma, clip_limit, tile_grid_size = params

    # Apply homomorphic filter
    filtered_image = hf.homomorphic_filter(reference_image, d0=d0, rh=rh, rl=rl)

    # Determine the number of BIMFs to denoise
    R = 1

    # Determine the beta value
    beta = 0.5

    # Reconstruct the image
    reconstructed_image = nf.denoise(BIMFs, energies, R, beta, filtered_image)

    # Apply gamma correction
    gamma_corrected_image = gamma_correction(reconstructed_image, gamma=gamma)

    # Apply CLAHE
    clahe_image = apply_clahe(gamma_corrected_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    # Calculate CII
    mask = np.ones_like(reference_image)
    cii = calculate_cii(clahe_image, reference_image, mask)

    # Calculate entropy
    entropy = calculate_entropy(clahe_image)

    # Calculate EME
    r, c = 4, 4
    eme = calculate_eme(clahe_image, r, c)

    evaluation_result = cii + entropy + eme
    return evaluation_result, cii, entropy, eme, params, clahe_image

# Use ThreadPoolExecutor to process parameter combinations concurrently
with ThreadPoolExecutor(8) as executor:
    futures = [executor.submit(process_parameters, params) for params in parameter_combinations]

    for future in as_completed(futures):
        evaluation_result, cii, entropy, eme, params, clahe_image = future.result()
        print(f"params {params}, score {evaluation_result}, cii {cii}, ent {entropy}, eme {eme}")
        if evaluation_result > best_cii + best_entropy + best_emr:
            best_cii = cii
            best_entropy = entropy
            best_emr = eme
            best_parameters = params
            best_image = clahe_image
# Save best image

filename = Path(fileIn).stem
filename = filename + "_processed_image.tiff"
cv2.imwrite(path + filename, best_image.astype(np.uint16))
resizeImage = cp.array(imResize(path + filename, size=(1024, 1024)))

filename = Path(filename).stem
filename = filename + "_1024.tiff"
filepath = path+filename

cv2.imwrite(filepath, resizeImage.get().astype(np.uint16))


# Compare best result image with reference image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(best_image, cmap='gray')
plt.title("Best Processed Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Show best parameters
print("Best Parameters:")
print(f"d0: {best_parameters[0]}")
print(f"rh: {best_parameters[1]}")
print(f"rl: {best_parameters[2]}")
print(f"gamma: {best_parameters[3]}")
print(f"clip_limit: {best_parameters[4]}")
print(f"tile_grid_size: {best_parameters[5]}")

