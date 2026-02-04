import cv2
import numpy as np

import gc

def homomorphic_filter(image, d0=30, rh=2.0, rl=0.5, c=1.0):
    if not isinstance(image, np.ndarray):
        img = image.get()
    else:
        img = image
    rows, cols = img.shape

    # Apply logarithmic transform
    log_image = np.log1p(img)
    
    # Perform Fourier transform
    dft = cv2.dft(log_image, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    dft_shift = np.fft.fftshift(dft)
    #print(dft_shift.shape)

    # Create high-frequency emphasis filter (HEF)
    u = np.arange(cols)
    v = np.arange(rows)
    u, v = np.meshgrid(u - rows / 2, v - cols / 2)
    d = np.sqrt(u**2 + v**2)
    h = (rh - rl) * (1 - np.exp(-c * (d**2 / d0**2))) + rl
    #print(h.shape)
    # Apply filter
    h = np.repeat(h[:, :, np.newaxis], 2, axis=2)
    
    dft_shift_filtered = dft_shift * h

    # Perform inverse Fourier transform
    dft_shift_filtered = np.fft.ifftshift(dft_shift_filtered)
    idft = cv2.idft(dft_shift_filtered)
    idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

    # Normalize the idft result to avoid overflow in expm1
    idft = cv2.normalize(idft, None, 0, 1, cv2.NORM_MINMAX)

    # Apply exponential transform
    exp_image = np.expm1(idft)

    # Handle NaN and Inf values
    exp_image = np.nan_to_num(exp_image, nan=0.0, posinf=65535.0, neginf=0.0)

    # Normalize the image to 0-255
    exp_image = cv2.normalize(exp_image, None, 0, 65535, cv2.NORM_MINMAX)
    exp_image = np.uint16(exp_image)

    del img, rows, cols
    del log_image, dft, dft_shift
    del u, v, d, h
    del dft_shift_filtered, idft
    gc.collect()
    
    return exp_image
