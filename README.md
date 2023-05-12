# 3D_tracing

using two csi cameras to get depth img in jetson nano/raspberry pi,the delay between two camera is lowwer than 1ms,which is acceptable errors.

this project only work in simple and regular target in moving.using BackgroundSubtractor to detect and 

fellow these steps to get your own tracing!
1.using’‘’csi_get_img.py'''or'''usb_get_img.py'''to get your pictures in two cameras.
2.using'''xxx_singleCalibrate.py'''to calibrate your cameras,the matrixs and distortion coeffs will be saved in a specific folder automatically
3.using'''xxx_stereoCalibrate.py'''to get R and T(matrix or vector between camera0 and camera1) 
4.'''getRoute.py'''to analyze and caculate your target points' 3D position.
