import cv2
import threading

def capture(camera_num):
    cap = cv2.VideoCapture(camera_num)
    while True:
        ret, frame = cap.read()
        if ret:
            # 在这里对图像帧进行处理
            cv2.imshow('Camera %d' % camera_num, frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    thread1 = threading.Thread(target=capture, args=(0,))
    thread2 = threading.Thread(target=capture, args=(2,))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    cv2.destroyAllWindows()
