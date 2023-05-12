import cv2
import numpy as np

# 设置CharuCo板参数
board = cv2.aruco.CharucoBoard_create(6, 9, 0.04, 0.02, cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250))
# 设置标定板参数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

pic_num = 10

# 获取标定板角点
def get_corners(img, board):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, board.dictionary)
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    return charuco_corners, charuco_ids

# 设置标定图像路径

# 初始化空数组
img_left_points = [] # 2D points
img_right_points = [] # 2D points
obj_points = [] # 3D points

# 遍历标定图像
for i in range(1,pic_num + 1):
    img_left  = cv2.imread(f'./img/charuco/left/{i}.jpg')
    img_right = cv2.imread(f'./img/charuco/right/{i}.jpg')
    corners_left ,  ids = get_corners(img_left, board)
    corners_right, ids = get_corners(img_right, board)
    if corners_left is not None and corners_right is not None and ids is not None:
        img_left_points.append(corners_left)
        img_right_points.append(corners_right)
        obj_points.append(board.chessboardCorners)