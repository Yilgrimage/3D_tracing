import numpy as np
import cv2

# Define the size of the calibration pattern
pattern_size = (11, 8)#为了保持旋转不变，行数必须是偶数，列数必须是奇数
square_size = 25.0
num_images = 10

id = 'right'

# Create a list of object points
object_points = np.zeros((np.prod(pattern_size), 3), np.float32)
object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)*square_size

# Create lists to store the object points and image points
object_points_list = []
image_points_list = []

# Capture a set of calibration images
for i in range(1,num_images + 1):
    # Load the image and convert to grayscale
    img  = cv2.imread(f'../img/zhang/{id}/{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners in the image
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    # If corners are found, add to object points and image points lists
    if ret:
        object_points_list.append(object_points)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
        image_points_list.append(corners)#improve the found corners' coordinate accuracy
        ####

        # img_temp = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        # cv2.imshow('img', img_temp)
        # cv2.waitKey(0)#press any key to show next image
        
        ####
# Calibrate the camera
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points_list, image_points_list, gray.shape[::-1], None, None)

# Save the camera matrix and distortion coefficients to a file

np.savez(f'../data/calibrate_data/{id}_calibration.npz', camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs)
print(f'finish calibration, saved as /data/calibrate_data/{id}_calibration.npz')