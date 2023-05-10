import numpy as np
import cv2

# 两个相机的内参和畸变
K1 = np.array([[725.849, 0, 644.456], [0, 724.791, 371.799], [0, 0, 1]])
K2 = np.array([[724.634, 0, 623.019], [0, 723.78, 375.482], [0, 0, 1]])
D1 = np.array([[-0.258825], [0.0377914], [-0.0014377], [0.000246389]])
D2 = np.array([[-0.278821], [0.0568747], [0.000729472], [0.000733384]])

# 两个相机之间的旋转矩阵和偏移
R = np.array([[0.99619, -0.0104595, 0.0869078], [0.0109791, 0.999888, 0.00828587], [-0.086862, -0.00889488, 0.996146]])
T = np.array([[-99.2612], [2.16175], [7.56518]])

# 已知点在两个相机上的对应关系

#pt1 = np.array([[279, 254], [292, 251], [276, 335], [292, 331]], dtype=np.float32)
#pt2 = np.array([[242, 252], [255, 248], [238, 332], [254, 329]], dtype=np.float32)
pt1 = np.array([[279,254]], dtype=np.float32)
pt2 = np.array([[242,252]], dtype=np.float32)

# 将像素坐标转换为归一化坐标
pt1_norm = cv2.undistortPoints(pt1.reshape(-1, 1, 2), K1, D1)
pt1_norm = np.concatenate(pt1_norm, axis=0).reshape(-1, 2)
pt2_norm = cv2.undistortPoints(pt2.reshape(-1, 1, 2), K2, D2)
pt2_norm = np.concatenate(pt2_norm, axis=0).reshape(-1, 2)

# 将归一化坐标转换为齐次坐标
pt1_homo = cv2.convertPointsToHomogeneous(pt1_norm)
pt2_homo = cv2.convertPointsToHomogeneous(pt2_norm)

# 计算对应点的外参矩阵
R1 = np.eye(3)
T1 = np.zeros((3, 1))
R2 = R
T2 = T

P1 = np.dot(K1,np.hstack((R1,T1)))
P2 = np.dot(K2,np.hstack((R2,T2)))

# 将齐次坐标还原为3D坐标
# pt3d = cv2.triangulatePoints(P1,P2,pt1_homo[0],pt2_homo[0])
pt3d = cv2.triangulatePoints(P1,P2,pt1_norm[0],pt2_norm[0])

pt3d /= pt3d[3]

# 显示结果
print('World coordinates:')
print(pt3d[:3])
