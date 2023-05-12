import numpy as np
import cv2
import glob

# 设置棋盘格的大小，单位为毫米
square_size = 20

# 设置棋盘格的宽度和高度
pattern_size = (11, 8)#为了保持旋转不变，行数必须是偶数，列数必须是奇数

# 准备棋盘格的3D坐标
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
# objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)*square_size

# 创建空列表，用于存储棋盘格角点的2D和3D坐标
objpoints = [] # 存储3D坐标
left_imgpoints = [] # 存储左侧相机的2D坐标
right_imgpoints = [] # 存储右侧相机的2D坐标

pic_num = 10

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for i in range(1,pic_num+1):
    #读取两个图片
    left_images = cv2.imread('./img/left/' + str(i) + '.jpg')
    right_images = cv2.imread('./img/right/' + str(i) + '.jpg')
    #灰度化
    left_gray = cv2.cvtColor(left_images, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_images, cv2.COLOR_BGR2GRAY)
    #寻找棋盘格角点
    ret_left, corners_left = cv2.findChessboardCorners(left_gray, pattern_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(right_gray, pattern_size, None)
    #如果找到了，添加到objpoints和imgpoints中
    if ret_left and ret_right:
        objpoints.append(objp)
        corners2_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
        left_imgpoints.append(corners2_left)
        corners2_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
        right_imgpoints.append(corners2_right)
    
# 使用棋盘格角点来计算相机参数
#ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, None, None,None, None,(1440,1920))

###
camera_left_data  = np.load('../data/calibrate_data/left_calibration.npz')
camera_right_data = np.load('../data/calibrate_data/right_calibration.npz')

mtx_left = camera_left_data['camera_matrix']
dist_left = camera_left_data['distortion_coeffs']

mtx_right = camera_right_data['camera_matrix']
dist_right = camera_right_data['distortion_coeffs']

ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, mtx_left, dist_left,mtx_right, dist_right,(1440,1920))

###
if ret:
    print('mtx_left =', mtx_left)
    print('dist_left =', dist_left)
    print('mtx_right =', mtx_right)
    print('dist_right =', dist_right)
    print('R =', R)
    print('T =', T)
    print('E =', E)
    print('F =', F)
    # 创建一个numpy数组来保存标定参数
    #data = np.array([ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F])

    # 保存标定数据到文件
    np.save('../data/calibrate_data/stereo_calibration.npz', ret = ret,mtx_left=mtx_left,dist_left = dist_left,
                            mtx_right = mtx_right,dist_right = dist_right, R = R,T = T,E = E,F = F)
else:
    print('error')

