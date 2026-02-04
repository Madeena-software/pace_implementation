import numpy as np
def gamma_correction(image, gamma=0.8):
    # Normalize the image to the range [0, 1]
    # img_normalized = image / 65535.0
    
    # Apply gamma correction
    img_gamma_corrected = np.power(image, gamma)
    
    # Scale back to the 16-bit range [0, 65535]
    #img_gamma_corrected = np.uint16(img_gamma_corrected * 65535)
    
    return img_gamma_corrected