import cv2
import numpy as np
import random
import cv2
import math
import time
import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random

remove_div_points = 8
length_filter = 30
random_points =   20 # int(0.75 *  length_filter ) -1 # num + 0.25 Length < Length 
mtx = np.genfromtxt("calib_mtx_webcam.csv")
dist = np.genfromtxt("calib_dist_webcam.csv")


def get_circle_center(p1, p2, p3):
    """
    计算三个点的外接圆圆心
    """
    temp = p2[0]**2 + p2[1]**2
    bc = (p1[0]**2 + p1[1]**2 - temp) / 2
    cd = (temp - p3[0]**2 - p3[1]**2) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1e-6:
      return None
    
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    return int(cx), int(cy)

def is_circle(contour,centers):
    """
    检查圆心分布是否接近，判断是否为圆形
    """
    n = remove_div_points
    threshold = 4

    # 将列表转换为 NumPy 数组
    centers = np.array(centers)
    # 计算均值中心
    mean_center = np.mean(centers, axis=0)
    # 计算每个点到均值中心的距离
    distances = np.linalg.norm(np.squeeze(centers - mean_center), axis=1)
    # 找到距离最大的 n 个点的索引
    indices_to_remove = np.argpartition(distances, -n)[-n:]
    # 创建一个布尔掩码，标记出需要保留的点
    mask = np.ones(centers.shape[0], dtype=bool)
    mask[indices_to_remove] = False
    # 使用掩码创建一个新的数组，不包含最大的 n 个值
    filtered_centers = centers[mask]

    #print('number of centers', len(filtered_centers))
    # 计算 去除散点的中心
    centers = np.array(filtered_centers)
    mean_center = np.mean(centers, axis=0)
    temp = np.squeeze(centers - mean_center)
    distances = (np.linalg.norm(np.squeeze(centers - mean_center), axis=1))
    
    # 计算半径
    Rs = (np.linalg.norm(np.squeeze(contour - mean_center), axis=1))
    R = np.mean(Rs)
    #print('div',np.mean(distances),'R',R,'mean center',mean_center)
    return (np.mean(distances) < threshold ) , mean_center ,R


if __name__ == '__main__':

    vid = cv2.VideoCapture(2)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 1280
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 720

    #video_path = 'tennis/1.mp4'
   # vid = cv2.VideoCapture(video_path)

    while True:

        ret, frame = vid.read()
        #frame = cv2.imread('tennis/1.jpg')

        #if not ret:
          #  vid = cv2.VideoCapture(video_path)
         #   ret, frame = vid.read()
            

        ts = time.time() # record time usage
        Original = frame.copy()
        cv2.imshow('Original',Original)
        # A  
        # 将图像从 BGR 转换为 HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 定义网球的颜色范围（根据网球的实际颜色调整）
        mask_1 = cv2.inRange(hsv, np.array([25, 70, 0]), np.array([60, 120, 255]))
        mask_2 = cv2.inRange(hsv, np.array([25, 20, 50]), np.array([60, 255, 160])) #backlighting(beiguang)
        mask_3 = cv2.inRange(hsv, np.array([8, 0, 220]), np.array([72, 100, 255])) #frontlighting(fanguang)

        # 创建遮罩，只保留在颜色范围内的部分 球
        mask = cv2.bitwise_or(mask_1, mask_2)
        mask = cv2.bitwise_or(mask, mask_3)
        # 蓝色地面，
        mask_ground = cv2.inRange(hsv, np.array([107, 69, 0]), np.array([126, 200, 255]))

        # 使用形态学操作去除噪声
        mask = cv2.erode(mask, None, iterations=1)
        cv2.imshow('erode',mask)
        tennis_dilated_mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=4)
        cv2.imshow('tennis_dilated_mask',tennis_dilated_mask)
        mask_ground = cv2.erode(mask_ground, np.ones((5, 5)), iterations=4)
        mask_ground = cv2.dilate(mask_ground, np.ones((15, 15), np.uint8), iterations=4)
        cv2.imshow('mask_ground',mask_ground)

        #B
        # 使用Canny边缘检测 并且和ground 颜色取交集
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)
        edges = cv2.Canny(blurred, 50, 150)
        edges_2 = cv2.Canny(gray, 50, 150)
        #edges = cv2.bitwise_and(edges, mask_ground)
        # 对 edges 进行膨胀操作
        edges_dilated = cv2.dilate(edges, None, iterations=1)
       
       #C
        # 找颜色和轮廓重合部分
        # 计算重叠部分
        intersection = cv2.bitwise_and(tennis_dilated_mask, edges)
        # 找到重叠部分的轮廓
        intersection_contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 保留最长的10个轮廓，并且保留大于 length_filter 的轮廓
        intersection_contours = sorted(intersection_contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:20]
        filtered_contours = [contour for contour in intersection_contours if len(contour) > length_filter]
        # 保留重叠的轮廓黑底图上
        result_image = np.zeros_like(frame)
        cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 3)
        #D
        valid_circles = []
        mean_centers = []
        Radius = []
        # 验证每个轮廓是否为圆形
        for contour in filtered_contours:
            centers = []
            for k in range(random_points):
                # 已知端点
                #length = len(contour)
                #point_1_idx = (k) 
                #point_2_idx = (int(length/4 + k)) 
                #point_3_idx = (int(length/8 + k/2)) 
                #p1, p2, p3 = contour[point_1_idx][0], contour[point_2_idx][0], contour[point_3_idx][0]
                # 随机采样法
                idx1, idx2, idx3 = np.random.choice(len(contour), 3, replace=True)
                p1, p2, p3 = contour[idx1][0], contour[idx2][0], contour[idx3][0]
                center = get_circle_center(p1, p2, p3)

                if center:
                    centers.append(center)
                    #cv2.circle(result_image, center, 1, (len(contour), len(contour), len(contour)), 2)
            if len(centers) > remove_div_points +1:
                check,m_center,R = is_circle(contour,centers)
            if check: 
                mean_centers.append(m_center)
                valid_circles.append(contour)
                Radius.append(R)

        #E
        # 初始化最大面积， 计算圆的最大的面积,找到面积最大的球
        max_area = 0
        max_contour = None
        for contour in valid_circles:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # 以圆心，半径，画出框
        for i in range(len(Radius)):
            Radius = np.array(Radius)
            max_R = np.max(Radius)

            if cv2.contourArea(valid_circles[i]) > 0.5 * max_area and max_R > 5:
                R_int = int(Radius[i])
                center = mean_centers[i]
                x, y = map(int, map(round, center))
                cv2.circle(frame, (x,y), 1, (0,0,255), 8)
                cv2.rectangle(frame, (x-R_int, y-R_int), (x + R_int, y + R_int), (0, 0, 255), 3)
                
                real_R = 6.5/200

                world_points = np.array([
                    [-real_R, 0, 0],      # left
                    [real_R, 0, 0],       # right
                    [0, real_R, 0],  # top
                    [0, -real_R, 0]        # bottom
                ], dtype=np.float32)
                pos_array = np.array([[x - R_int, y], [x + R_int, y], [x, y + R_int], [x, y-R_int]], dtype=np.float32)
                print('pos_array\n', pos_array)
                success, rvec, tvec = cv2.solvePnP(world_points, pos_array, mtx, dist)
                distance = np.linalg.norm(tvec)
                print('distance',distance)
            
                print('circle number : ', i+1 , 'position ' , x-320, y-240, 'R',R_int)

        
    


        
        #print('time used all', time.time() - ts)
        x_offset = 800
        y_offset = 600
        # 显示结果图像
        cv2.imshow('erode' ,mask)
        cv2.moveWindow('erode', x_offset * 0, y_offset * 0)
        cv2.imshow('color mask' ,tennis_dilated_mask)
        cv2.moveWindow('color mask', x_offset * 0, y_offset * 1)
        cv2.imshow('ground mask' ,mask_ground)
        cv2.moveWindow('ground mask', x_offset * 0, y_offset * 2)
        cv2.imshow('edge capture',edges)
        cv2.moveWindow('edge capture', x_offset * 1, y_offset * 0)
        cv2.imshow('contour after color comb',result_image)
        cv2.moveWindow('contour after color comb', x_offset * 1, y_offset * 1)
        cv2.imshow('tennis',frame)
        cv2.moveWindow('tennis', x_offset * 1, y_offset * 2)
        cv2.imshow('Original',Original)
        cv2.moveWindow('Original', x_offset * 2-150, y_offset * 2)
        
        


        time.sleep(0.01)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

