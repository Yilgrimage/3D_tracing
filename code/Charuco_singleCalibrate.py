import cv2
import numpy as np
import glob
# 设置CharuCo板参数
board = cv2.aruco.CharucoBoard((6, 9), 40, 31, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))

# 设置标定板参数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

# 获取标定板角点
def get_corners(img, board):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, board.dictionary)
    if corners is not None and len(corners) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
            return charuco_corners, charuco_ids
    return None, None


# 初始化空数组
img_points = [] # 2D points
obj_points = [] # 3D points

pic_num = 10

id = 'right'

# 遍历标定图像
for i in range(1,pic_num + 1):
    img = cv2.imread(f'../img/charuco/{id}/{i}.jpg')
    corners, ids = get_corners(img, board)
    if corners is not None and ids is not None:
        img_points.append(corners)
        obj_points.append(board.chessboardCorners)
        cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)
        cv2.imshow('img', img)
        cv2.waitKey(0)

# 标定相机
ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(obj_points, img_points, board, img.shape[:2], None, None)

print('mtx =', camera_matrix)
print('dist =', distortion_coefficients)
print('R =', rvecs)
print('T =', tvecs)

# 保存标定结果
np.savez('../data/calibrate_data/{id}_calibration.npz', camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)
