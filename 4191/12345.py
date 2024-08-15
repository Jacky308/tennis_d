import cv2
import numpy as np

def nothing(x):
    pass

# 创建一个窗口
cv2.namedWindow('Color Filter')

# 创建滑动条用于调整颜色范围
cv2.createTrackbar('Lower H', 'Color Filter', 0, 179, nothing)
cv2.createTrackbar('Lower S', 'Color Filter', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Color Filter', 0, 255, nothing)
cv2.createTrackbar('Upper H', 'Color Filter', 179, 179, nothing)
cv2.createTrackbar('Upper S', 'Color Filter', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Color Filter', 255, 255, nothing)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头图像
    frame = cv2.imread('tennis/13.jpg')
    if not 1:
        break

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 获取滑动条位置
    lower_h = cv2.getTrackbarPos('Lower H', 'Color Filter')
    lower_s = cv2.getTrackbarPos('Lower S', 'Color Filter')
    lower_v = cv2.getTrackbarPos('Lower V', 'Color Filter')
    upper_h = cv2.getTrackbarPos('Upper H', 'Color Filter')
    upper_s = cv2.getTrackbarPos('Upper S', 'Color Filter')
    upper_v = cv2.getTrackbarPos('Upper V', 'Color Filter')

    # 设置颜色范围
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # 颜色过滤
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 显示结果
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Color Filter', result)

    # 按下ESC键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
