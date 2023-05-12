from __future__ import print_function
import cv2 as cv
import numpy as np

#用于提取运动中圆环的几何中心，主要是将圆环视为圆盘，通过两个相机拍摄所得的‘圆盘中心’来确定‘中心的轨迹’

history = 200  # 训练帧数
dist2Threshold = 2000.0  # 阈值距离
detectShadows = False  # 不检测阴影
backSub = cv.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist2Threshold, detectShadows=detectShadows)


#capture = cv.VideoCapture(0)
capture = cv.VideoCapture('video_2.mp4')

capture.set(cv.CAP_PROP_AUTO_WB, 0)

dots_array = []

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    fgMask_row = np.copy(fgMask)

    fgMask = cv.medianBlur(fgMask, 5)
    fgMask = cv.threshold(fgMask, 10, 255, cv.THRESH_BINARY)[1]

    morph_kernel = np.ones((10, 10), np.uint8)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, morph_kernel)


    contours, hierarchy = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
    contour_img = np.zeros_like(frame)

    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    if max_contour is not None:

######################################################
# #使用ROI
#         x,y,w,h = cv.boundingRect( max_contour)  # 获取第一个轮廓的外接矩形
        
#         dialate_rate = 0.2
#         x1 = x-dialate_rate*w
#         x2 = x+(1+dialate_rate)*w
#         y1 = y-dialate_rate*h
#         y2 = y+(1+dialate_rate)*h
#         img_roi = np.zeros_like(fgMask)
#         roi = fgMask[y1:y2, x1:x2]
#         img_roi[y1:y2, x1:x2] = roi 

#         img_roi = cv.threshold(fgMask, 10, 255, cv.THRESH_BINARY)[1]
#         img_roi = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, np.ones((10, 10), np.uint8))
#         contours, hierarchy = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         # 绘制轮廓
#         contour_img = np.zeros_like(frame)

#         max_contour = None
#         max_area = 0
#         for contour in contours:
#             area = cv.contourArea(contour)
#             if area > max_area:
#                 max_area = area
#                 max_contour = contour

#########################################################

        hull = cv.convexHull(max_contour)
        M = cv.moments(hull)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        cv.drawContours(contour_img, [hull], 0, (0, 255, 0), 2)
        cv.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)
    dots_array.append((cx,cy))
    # 显示图像
    cv.imshow('Frame', frame)
    # cv.imshow('FG Mask row', fgMask_row)
    # cv.imshow('FG Mask', fgMask)
    cv.imshow('Contours', contour_img)

    keyboard = cv.waitKey(200)
    if keyboard == 'q' or keyboard == 27:
        break
capture.release()