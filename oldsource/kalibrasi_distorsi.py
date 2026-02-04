import cv2
import numpy as np
from pathlib import Path
import sys

def get_distortion_parameters(calibration_image_path, parameters_file, pattern_size=(44,35), roi_crop=(0, 0, 4096, 3000)):
    # Prepare object points based on the real-world coordinates of the dot matrix points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    img = cv2.imread(calibration_image_path, cv2.IMREAD_GRAYSCALE)
    ret, centers = cv2.findCirclesGrid(img, pattern_size, cv2.CALIB_CB_SYMMETRIC_GRID)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        imgpoints.append(centers)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img.shape[::-1], None, None
    )

    np.savez(parameters_file, 
             mtx=mtx, 
             dist=dist, 
             rvecs=rvecs, 
             tvecs=tvecs, 
             roi=roi_crop)
    param_file = Path(parameters_file)

    print('Distortion Parameters')
    print('mtx = ',mtx)
    print('dsit = ',dist)
    print('rvecs = ',rvecs)
    print('tvecs = ',tvecs)
    print('roi = ',roi_crop)

    if param_file.exists():
        return True
    else:
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python kalibrasi_distorsi.py <calibration_image_path> <parameters_file> [row col] [x y w h]")
        sys.exit(1)
    image_path = sys.argv[1]
    param_file = sys.argv[2]
    pat_size = tuple(sys.argv[3],sys.argv[4]) if len(sys.argv) > 4 else None
    roi = tuple(sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8]) if len(sys.argv) > 8 else None

    if pat_size is None and roi is None:
        if get_distortion_parameters(calibration_image_path=image_path,
                                 parameters_file=param_file):
            print('Parameters Successfully save to : '+ param_file)
        else:
            print('Error Saving Parameters File')
    elif roi is None:
        if get_distortion_parameters(calibration_image_path=image_path,
                                 parameters_file=param_file, pattern_size=pat_size):
            print('Parameters Successfully save to : '+ param_file)
        else:
            print('Error Saving Parameters File')
    else:
        get_distortion_parameters(calibration_image_path=image_path,
                                 parameters_file=param_file,
                                 pattern_size=pat_size,
                                 roi_crop=roi)
        print('Parameters Successfully save to : '+ param_file)

if __name__ == "__main__":
    main()