import cupy as cp
from  cupyx.scipy.ndimage import zoom
import cv2
import pathlib as Path


def resize(image, new_width):
    # Calculate the new dimensions
    height, width = image.shape[:2]
    width_percent = new_width / float(width)
    new_height = int(height * width_percent)

    # Resize the image using cupyx.scipy.ndimage.zoom
    zoom_factors = (new_height / height, new_width / width, 1) if len(image.shape) == 3 else (new_height / height, new_width / width)
    resized_image = zoom(cp.array(image), zoom_factors, order=1)
    return resized_image

def resize_image(input_image, new_width, input_path=None, output_path=None):
    #check if input_image is not None
    if input_image is not None:
        #check if input_image is a cupy array
        if isinstance(input_image, cp.ndarray):
            image = input_image
        else:
            image = cp.array(input_image)
        return resize(image, new_width)
    else:
        # Load the image
        image = cv2.imread(input_path, -1)
        # Resize the image
        resized_image = resize(image, new_width)     
        # Convert back to numpy array and save the image
        resized_image = cp.asnumpy(resized_image)
        cv2.imwrite(output_path, resized_image)
        out_file = Path(output_path)
        if out_file.exists():
            return True
        else:
            return False