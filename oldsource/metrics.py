import numpy as np
import cv2


def calculate_contrast(image, mask):
    """
    Calculate the contrast of a region in the image defined by the mask.

    Parameters:
    image (np.ndarray): Grayscale image.
    mask (np.ndarray): Binary mask defining the region of interest.

    Returns:
    float: Contrast value.
    """
    foreground = image[mask == 1]
    background = image[mask == 0]

    # print("fore", foreground)
    # print("back", background)

    # if len(foreground) == 0 or len(background) == 0:
    #     return 0.0  # Return 0 contrast if there are no valid pixels

    X_f = np.mean(foreground)
    X_b = np.mean(background)

    if len(background) == 0:
        X_b = 0

    if X_f + X_b == 0:
        return 0.0  # Avoid division by zero

    contrast = (X_f - X_b) / (X_f + X_b)
    return contrast


def calculate_cii(processed_image, reference_image, mask):
    """
    Calculate the Contrast Improvement Index (CII).

    Parameters:
    processed_image (np.ndarray): Processed grayscale image.
    reference_image (np.ndarray): Reference (original) grayscale image.
    mask (np.ndarray): Binary mask defining the region of interest.

    Returns:
    float: CII value.
    """
    C_processed = calculate_contrast(processed_image, mask)
    C_reference = calculate_contrast(reference_image, mask)

    # print(C_processed)
    # print(C_reference)

    CII = C_processed / C_reference
    return CII

def calculate_entropy(image):
    """
    Calculate the entropy of an image.

    Parameters:
    image (np.ndarray): Grayscale image.

    Returns:
    float: Entropy value.
    """
    # Hitung histogram gambar
    hist = cv2.calcHist([image], [0], None, [65535], [0, 65535])

    # Normalisasi histogram sehingga jumlahnya menjadi 1
    hist = hist / hist.sum()

    # Hitung entropy
    entropy = -np.sum(
        hist * np.log(hist + 1e-7)
    )  # Tambahkan 1e-7 untuk menghindari log(0)

    return entropy


def calculate_eme(image, r, c, epsilon=0.0001):
    """
    Calculate the Effective Measure of Enhancement (EME) of an image.

    Parameters:
    image (np.ndarray): Grayscale image.
    r (int): Number of rows to split the image into.
    c (int): Number of columns to split the image into.
    epsilon (float): Small constant to avoid division by zero.

    Returns:
    float: EME value.
    """
    height, width = image.shape
    block_height = height // r
    block_width = width // c

    eme = 0.0
    for i in range(r):
        for j in range(c):
            # Extract the block
            block = image[
                i * block_height : (i + 1) * block_height,
                j * block_width : (j + 1) * block_width,
            ]

            # Calculate the maximum and minimum intensity levels in the block
            I_max = np.max(block)
            I_min = np.min(block)

            if I_min + epsilon == 0:
                continue  # Skip this block to avoid division by zero

            # Calculate the contrast ratio for the block
            CR = I_max / (I_min + epsilon)

            # Update the EME value
            eme += 20 * np.log(CR)

    # Normalize the EME value by the number of blocks
    eme /= r * c

    return eme


# # Contoh penggunaan
# # Misalkan kita memiliki gambar yang diproses
# processed_image = cv2.imread("processed_image.png", cv2.IMREAD_GRAYSCALE)

# # Hitung EME dengan membagi gambar menjadi 4x4 blok
# r, c = 4, 4
# eme = calculate_eme(processed_image, r, c)

# # Tampilkan hasil
# print("Effective Measure of Enhancement (EME):", eme)