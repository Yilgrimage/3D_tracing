from __future__ import print_function
import cv2 as cv
import numpy as np

#用于提取运动中圆环的几何中心，主要是将圆环视为圆盘，通过两个相机拍摄所得的‘圆盘中心’来确定‘中心的轨迹’

def preprocess(fgMask):
    #fgMask = backSub.apply(frame)
    #fgMask_row = np.copy(fgMask)

    fgMask = cv.medianBlur(fgMask, 5)
    fgMask = cv.threshold(fgMask, 10, 255, cv.THRESH_BINARY)[1]

    morph_kernel = np.ones((10, 10), np.uint8)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, morph_kernel)
    return fgMask

def get_center(contour_img):
    contours, hierarchy = cv.findContours(contour_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    contour_img = np.zeros_like(contour_img)

    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    if max_contour is not None:
        hull = cv.convexHull(max_contour)
        M = cv.moments(hull)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        cv.drawContours(contour_img, [hull], 0, (0, 255, 0), 2)
        cv.circle(contour_img, (cx, cy), 2, (0, 0, 255), 2)
        #cv.imshow('contour_img', contour_img)
        return cx, cy, contour_img
    else:
        return None, None,contour_img

def get3Dpoint(x1, y1, x2, y2,K1,D1,K2,D2,R,T):
    pt1 = np.array([[x1,y1]], dtype=np.float32)
    pt2 = np.array([[x2,y2]], dtype=np.float32)
    # 将像素坐标转换为归一化坐标
    pt1_norm = cv.undistortPoints(pt1.reshape(-1, 1, 2), K1, D1)
    pt1_norm = np.concatenate(pt1_norm, axis=0).reshape(-1, 2)
    pt2_norm = cv.undistortPoints(pt2.reshape(-1, 1, 2), K2, D2)
    pt2_norm = np.concatenate(pt2_norm, axis=0).reshape(-1, 2)

    # 将归一化坐标转换为齐次坐标
    pt1_homo = cv.convertPointsToHomogeneous(pt1_norm)
    pt2_homo = cv.convertPointsToHomogeneous(pt2_norm)

    R1 = np.eye(3)
    T1 = np.zeros((3, 1))
    R2 = R
    T2 = T

    P1 = np.dot(K1,np.hstack((R1,T1)))
    P2 = np.dot(K2,np.hstack((R2,T2)))

    # 将齐次坐标还原为3D坐标
    # pt3d = cv.triangulatePoints(P1,P2,pt1_homo[0],pt2_homo[0])
    pt3d = cv.triangulatePoints(P1,P2,pt1_norm[0],pt2_norm[0])#返回的点是相对于cam1为原点的三维坐标
    pt3d /= pt3d[3]

    return pt3d[:3]

history = 200  # 训练帧数
dist2Threshold = 2000.0  # 阈值距离
detectShadows = False  # 不检测阴影
backSub = cv.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist2Threshold, detectShadows=detectShadows)

capture_left  = cv.VideoCapture('../video/video_left.mp4')
capture_right = cv.VideoCapture('../video/video_right.mp4')
capture_left.set(cv.CAP_PROP_AUTO_WB, 0)
capture_right.set(cv.CAP_PROP_AUTO_WB,0)

stereo_data = np.load('../data/calibrate_data/stereo_calibration.npz')

# 两个相机的内参和畸变
K1 = stereo_data['mtx_left']
K2 = stereo_data['mtx_right']
D1 = stereo_data['dist_left']
D2 = stereo_data['dist_right']

# 两个相机之间的旋转矩阵和偏移
R = stereo_data['R']
T = stereo_data['T']

dots_array = []

while True:
    _, frame_left = capture_left.read()
    _, frame_right = capture_right.read()

    if frame_left or frame_right is None:
        break
    
    fgMask_left = backSub.apply(frame_left)
    fgMask_right = backSub.apply(frame_right)

    #fgMask_left_raw = np.copy(fgMask_left)
    #fgMask_right_raw = np.copy(fgMask_right)

    #图像预处理
    fgMask_left = preprocess(fgMask_left)
    fgMask_right = preprocess(fgMask_right)

    cx_left, cy_left,contour_img_left     = get_center(fgMask_left)
    cx_right, cy_right, contour_img_right = get_center(fgMask_right)

    cv.imshow('sub',np.hstack((fgMask_left, fgMask_right)))
    cv.imshow('img', np.hstack((contour_img_left, contour_img_right)))
    if cx_left is not None and cx_right is not None:
        #dots_array.append([cx_left, cy_left, cx_right, cy_right])
        x,y,z = get3Dpoint(cx_left, cy_left, cx_right, cy_right,K1,D1,K2,D2,R,T)
        print('dots:'+ x,y,z)
    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break
capture_left.release()
capture_right.release()
cv.destroyAllWindows()