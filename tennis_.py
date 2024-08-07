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

#matplotlib nbagg


remove_div_points = 10
length_filter = 80
random_points = int(0.75 *  length_filter ) -1 # num + 0.25 Length < Length 
a,b,c,d,e = 1,1,1,1,1


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
    distances = (np.linalg.norm(np.squeeze(centers - mean_center), axis=1))
    
    # 计算半径
    Rs = (np.linalg.norm(np.squeeze(contour - mean_center), axis=1))
    R = np.mean(Rs)
    threshold = 7
    #if np.mean(distances) < threshold : 
        #print('R',R )
    #if np.mean(distances) < threshold:
        #print('div', np.mean(distances) , 'R',R ,'threshold',threshold )
    return (np.mean(distances) < threshold ) , mean_center ,R

def is_circle_2(contour,centers):
    """
    检查圆心分布是否接近，判断是否为圆形
    """
    mean_center = np.mean(centers, axis=0)
    diff = np.squeeze(contour - mean_center)
    Rs = ((np.linalg.norm(diff, axis=1)))
    print('diff',diff[0:5],'contour',contour[0:5],'mean_center',mean_center)
    div = np.var(np.sort(Rs)[0:-10])
    R = np.mean(Rs)
    threshold = 5
    print('div',div,'R',R,'mean center',mean_center)
    return (div < threshold ) , mean_center ,R



if __name__ == '__main__':

    vid = cv2.VideoCapture(2)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 1280
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 720



    while True:
        

        ret, frame = vid.read()
        # 读取示例图像
        #frame = cv2.imread('ball.png')  # 替换为你自己的图片路径

        ts = time.time()

        if a == 1:
            
            # 将图像从 BGR 转换为 HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 定义网球的颜色范围（根据网球的实际颜色调整）
            lower_range_1 = np.array([25, 70, 0])
            upper_range_1 = np.array([60, 120, 255])
            mask_1 = cv2.inRange(hsv, lower_range_1, upper_range_1)

            lower_range_2 = np.array([25, 20, 50])#backlighting(beiguang)
            upper_range_2 = np.array([60, 255, 160])
            mask_2 = cv2.inRange(hsv, lower_range_2, upper_range_2)

            lower_range_3 = np.array([0, 0, 220])#frontlighting(fanguang)
            upper_range_3 = np.array([72, 100, 255])
            mask_3 = cv2.inRange(hsv, lower_range_3, upper_range_3)


            # 创建遮罩，只保留在颜色范围内的部分
            mask = cv2.bitwise_or(mask_1, mask_2)
            mask = cv2.bitwise_or(mask, mask_3)

            # 使用形态学操作去除噪声
            mask = cv2.erode(mask, None, iterations=4)
            mask = cv2.dilate(mask, None, iterations=4)
            dilated_mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=4)


        if b ==1:

            # 使用Canny边缘检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 2)
            edges = cv2.Canny(blurred, 50, 150)


            # 对 edges 进行膨胀操作
            edges_dilated = cv2.dilate(edges, None, iterations=1)

            # 寻找轮廓
            contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 为每个轮廓生成一个随机颜色
            colors = [tuple(random.choices(range(256), k=3)) for _ in range(len(contours))]
            # 创建一个黑底图像
            black_image = np.zeros_like(frame)
            # 绘制每个过滤后的轮廓在黑底图上
            for contour, color in zip(contours, colors):
                cv2.drawContours(black_image, [contour], -1, color, 2)


        if c == 1:

            # 找颜色和轮廓重合部分
            # 创建一个空白掩码图像
            contour_mask = np.zeros_like(dilated_mask)
            # 在掩码图像上绘制 contours
            for contour in contours:
                cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            # 计算重叠部分
            intersection = cv2.bitwise_and(dilated_mask, contour_mask)
            # 找到重叠部分的轮廓
            intersection_contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 保留重叠的轮廓黑底图上
            result_image = np.zeros_like(frame)
            filtered_contours = [contour for contour in intersection_contours if len(contour) > length_filter]
            #for k in intersection_contours: 
                #print('len_ intersection_contours',len(k))
            cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)
            

            
            valid_circles = []
            mean_centers = []
            Radius = []
            #print('len filtered_contours', len(filtered_contours))
        
        if d == 1:

            # 验证每个轮廓是否为圆形
            for contour in filtered_contours:
                centers = []
                for k in range(random_points):
                    # 随机采样法
                    #idx1, idx2, idx3 = np.random.choice(len(contour), 3, replace=True)
                    #p1, p2, p3 = contour[idx1][0], contour[idx2][0], contour[idx3][0]
                    # 已知端点
                    length = len(contour)
                    point_1_idx = (k) 
                    point_2_idx = (int(length/4 + k)) 
                    point_3_idx = (int(length/8 + k/2)) 
                    p1, p2, p3 = contour[point_1_idx][0], contour[point_2_idx][0], contour[point_3_idx][0]
                    
                    center = get_circle_center(p1, p2, p3)

                    if center:
                        centers.append(center)
                        cv2.circle(result_image, center, 1, (len(contour), len(contour), len(contour)), 2)
                if len(centers) > remove_div_points :
                    check,m_center,R = is_circle(contour,centers)
                if check: 
                    mean_centers.append(m_center)
                    valid_circles.append(contour)
                    Radius.append(R)



        if e == 1:

            #print('m_center', m_center)
            circle_img = np.zeros_like(frame)
            for i in range(len(valid_circles)):
                # 将识别出的圆形轮廓标记在原图上
                contour = valid_circles[i]
                cv2.drawContours(circle_img, [contour], -1, (0, 255, 0), 4)


            #print('Radius',Radius)
            # 以圆心，半径，画出框
            for i in range(len(Radius)):
                Radius = np.array(Radius)
                max_R = np.max(Radius)

                if Radius[i] > 0.5 * max_R and max_R > 10:
                    R_int = int(Radius[i])
                    center = mean_centers[i]
                    x, y = map(int, map(round, center))
                    cv2.circle(frame, (x,y), 1, (len(contour), len(contour), len(contour)), 8)
                    cv2.rectangle(frame, (x-R_int, y-R_int), (x + R_int, y + R_int), (255, 0, 0), 5)

        print('time used all', time.time() - ts)
        # 显示结果图像
        cv2.imshow('erode' ,mask)
        cv2.imshow('color mask' ,dilated_mask)
        cv2.imshow('edge capture',edges)
        #cv2.imshow('contour',black_image)
        #cv2.imshow('detect as circle',circle_img)
        #cv2.imshow('contour after color comb',result_image)
        #cv2.imshow('tennis',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

