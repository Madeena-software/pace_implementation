import cv2
import numpy as np

def denoise(bimfs, energies, filtered_residual, R = 1, beta = 0.5):
    # Sort BIMFs based on energy
    sorted_indices = np.argsort(energies)

    # Denoise R components with the lowest energy
    denoised_bimfs = []
    for i in range(int(R)):
        index = sorted_indices[i]
        denoised_bimfs.append(cv2.bilateralFilter(bimfs[index].get().astype(np.float32), 5, 75, 75))   

    # Combine denoised BIMFs and original BIMFs
    I_E = np.sum(denoised_bimfs, axis=0)
    for j in range(int(R), len(bimfs)):
        index = sorted_indices[j]
        I_E += np.array(bimfs[index].get())

    # Reconstruct the image by adding the filtered residual
    I_L = I_E + beta * filtered_residual
    return I_L