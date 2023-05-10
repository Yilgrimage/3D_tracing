import cv2
import numpy as np
#import serial
import sys
sys.path.append('/home/yilgrimage/opencv')

ball_color = 'red'

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

def parity_check(data):
    parity = 1
    for i in range(7):
        if (data >> i) & 1:
            parity = ~parity
    return (data & 0b11111110) | (parity & 1)

#uart = serial.Serial('/dev/ttyTHS1', 115200)
#seven bits for data, one for parity check
#|0000| data;  |000| id;  |0| check
uart_data = 0b00000000

#open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")

#main loop
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)                     # 高斯模糊
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)                 # 转化成HSV图像
            erode_hsv = cv2.erode(hsv, None, iterations=2)                   # 腐蚀 粗的变细
            inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
            cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if cnts is not None and len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                center = np.mean(box,axis = 0)
                
                temp = np.int0(center[0]/40) & 0b1111   # Extract the 4-bit value and ensure it fits within 0-15
                uart_data = (uart_data & 0b00001111) | (temp << 4)  # Clear bits 1-4 and set them to the shifted value
                #uart.write(bytes([parity_check(uart_data)]))
                
                cnt_color = (255,0,255)
                cv2.drawContours(frame, [box], -1, cnt_color, 2)
            cv2.imshow('camera', frame)
            
            cv2.waitKey(1)
        else:
            print("无画面")
    else:
        print("无法读取摄像头！")

cap.release()
cv2.destroyAllWindows()