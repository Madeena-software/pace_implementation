import cv2
import numpy as np
import pathlib as Path

def undistort(mtx, dist, image):
    # Undistort the image
    h, w = image.shape[:2]
    newcameramtx, roi_rotate = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, mtx, dist, None, newcameramtx)
    undistorted_image = crop_image(undistorted_image, roi_rotate)
    return undistorted_image

def crop_image(image, roi):
    x, y, w, h = roi
    crop_image = image[y : y + h, x : x + w]
    return crop_image

def calibrate_spasial(parameters_path, image, image_path = None, result_path=None):
    # Load the calibration parameters
    with np.load(parameters_path) as params:
        mtx, dist, rvecs, tvecs, roi = [params[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs','roi')]
    if image is not None:
        if not isinstance(image, np.ndarray):
            img = image.get()
            img = undistort(mtx,dist,img)
            img = crop_image(img, roi)
            return img
        else:
            img = image
            img = undistort(mtx,dist,img)
            img = crop_image(img, roi)
            return img
    else:
        img = cv2.imread(image_path, -1)
        img = undistort(mtx,dist,img)
        img = crop_image(img, roi)
        cv2.imwrite(result_path, img)
        out_file = Path(result_path)
        if out_file.exists():
            return True
        else:
            return False