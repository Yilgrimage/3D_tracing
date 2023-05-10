import cv2 as cv

cap1 = cv.VideoCapture(2,cv.CAP_V4L2)
cap2 = cv.VideoCapture(0,cv.CAP_V4L2)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1:
        cv.imshow('frame1', frame1)
    else:
        print('camera1 not found')
    if ret2:
        cv.imshow('frame2', frame2)
    else:
        print('camera2 not found')
    key = cv.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('p'):
        cv.imwrite('left.jpg', frame1)
        cv.imwrite('right.jpg', frame2)
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()