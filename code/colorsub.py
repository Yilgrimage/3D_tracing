import cv2 as cv
import numpy as np
target_color = 'red'

color_dist = {'black': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 46])},
              'gray': {'Lower': np.array([0, 0, 46]), 'Upper': np.array([180, 43, 220])},
              'white': {'Lower': np.array([0, 0, 221]), 'Upper': np.array([180, 30, 255])},
              'orange': {'Lower': np.array([11, 43, 46]), 'Upper': np.array([25, 255, 255])},
              'yellow': {'Lower': np.array([26, 43, 46]), 'Upper': np.array([34, 255, 255])},
              'green': {'Lower': np.array([35, 43, 46]), 'Upper': np.array([77, 255, 255])},
              'pink': {'Lower': np.array([78, 43, 46]), 'Upper': np.array([99, 255, 255])},
              'purple': {'Lower': np.array([100, 43, 46]), 'Upper': np.array([124, 255, 255])},
              'red': {'Lower': np.array([125, 43, 46]), 'Upper': np.array([180, 255, 255])},
              'green': {'Lower': np.array([35, 43, 46]), 'Upper': np.array([77, 255, 255])},
              'brown': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 46])},
              'blue': {'Lower': np.array([100, 43, 46]), 'Upper': np.array([124, 255, 255])},
              }


cap = cv.VideoCapture('video_0.mp4')
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    gs_frame = cv.GaussianBlur(frame, (5, 5), 0)                     # 高斯模糊
    hsv = cv.cvtColor(gs_frame, cv.COLOR_BGR2HSV)                 # 转化成HSV图像
    erode_hsv = cv.erode(hsv, None, iterations=2)                   # 腐蚀 粗的变细
    inRange_hsv = cv.inRange(erode_hsv, color_dist[target_color]['Lower'], color_dist[target_color]['Upper'])
    cv.imshow('camera', inRange_hsv )      
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()