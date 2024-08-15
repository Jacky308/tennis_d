import cv2
import numpy as np

class tennis_detctor:
    def __init__(self, length_filter=30, remove_div_points=8,random_points=20,threshold=4):
        
        self.length_filter = length_filter # 30 
        self.remove_div_points = remove_div_points #8
        self.random_points =  random_points # 20
        self.threshold = threshold # 4
        
    def detect(self,frame):
        
        # 检查颜色范围1，找到网球颜色区域和地面颜色区域
        mask_ground = self.color_filter(frame,'ground')
        maks_tennis = self.color_filter(frame,'tennis')
        # 使用Canny 检测轮廓
        edges = self.Canny_edge(frame)
        edges = cv2.bitwise_and(edges, mask_ground)
        intersection = cv2.bitwise_and(maks_tennis, edges)
        # 找到重叠部分的轮廓
        intersection_contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 保留最长的10个轮廓，并且保留大于 length_filter 的轮廓
        intersection_contours = sorted(intersection_contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:20]
        filtered_contours = [contour for contour in intersection_contours if len(contour) > self.length_filter]
        # 检查轮廓是否是圆形，返回符合的⚪
        valid_circles,mean_centers,Radius = self.check_circles_in_filtered_contours(filtered_contours)
        
        # 初始化最大面积， 计算圆的最大的面积,找到面积最大的球
        tennis_list_center = []
        tennis_R = []
        max_area = 0
        for contour in valid_circles:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
        # 记下符合的圆形
        for i in range(len(Radius)):
            if cv2.contourArea(valid_circles[i]) > 0.5 * max_area and np.max(Radius) > 5:
                x, y = map(int, map(round, mean_centers[i]))
                tennis_list_center.append([x,y])
                tennis_R.append(int(Radius[i]))
        return tennis_list_center,tennis_R
        
    def color_filter(self,frame,obj):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if obj == 'tennis':
            # 定义网球的颜色范围（根据网球的实际颜色调整）
            mask_1 = cv2.inRange(hsv, np.array([25, 70, 0]), np.array([60, 120, 255]))
            mask_2 = cv2.inRange(hsv, np.array([25, 20, 50]), np.array([60, 255, 160])) #backlighting(beiguang)
            mask_3 = cv2.inRange(hsv, np.array([8, 0, 220]), np.array([72, 100, 255])) #frontlighting(fanguang)
            # 创建遮罩，只保留在颜色范围内的部分 球
            mask = cv2.bitwise_or(mask_1, mask_2)
            mask = cv2.bitwise_or(mask, mask_3)
            # 使用形态学操作去除噪声
            mask = cv2.erode(mask, None, iterations=1)
            obj_dilated_mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=4)
        elif obj == 'ground':
            # 蓝色地面，
            mask_ground = cv2.inRange(hsv, np.array([107, 69, 0]), np.array([126, 200, 255]))
            # 使用形态学操作去除噪声
            mask_ground = cv2.erode(mask_ground, np.ones((5, 5)), iterations=4)
            obj_dilated_mask = cv2.dilate(mask_ground, np.ones((15, 15), np.uint8), iterations=4)
        return obj_dilated_mask
    
    def Canny_edge(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)
        edges = cv2.Canny(blurred, 50, 150)
        edges_dilated = cv2.dilate(edges, None, iterations=1)
        return edges
    
    def check_circles_in_filtered_contours(self, filtered_contours):
        valid_circles = []
        mean_centers = []
        Radius = []
        check = None
        # 验证每个轮廓是否为圆形
        for contour in filtered_contours:
            centers = []
            for k in range(self.random_points):
                # 随机采样法
                idx1, idx2, idx3 = np.random.choice(len(contour), 3, replace=True)
                p1, p2, p3 = contour[idx1][0], contour[idx2][0], contour[idx3][0]
                center = self.get_circle_center(p1, p2, p3)

                if center:
                    centers.append(center)
                    #cv2.circle(result_image, center, 1, (len(contour), len(contour), len(contour)), 2)
            if len(centers) > self.remove_div_points +1:
                check,m_center,R = self.is_circle(contour,centers)
            if check: 
                mean_centers.append(m_center)
                valid_circles.append(contour)
                Radius.append(R)
        return valid_circles,mean_centers,Radius
    
    def get_circle_center(self,p1, p2, p3):
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

    def is_circle(self,contour,centers):
        """
        检查圆心分布是否接近，判断是否为圆形
        """
        # 将列表转换为 NumPy 数组, 计算均值中心, 计算每个点到均值中心的距离,找到距离最大的 n 个点的索引
        # 创建一个布尔掩码，标记出需要保留的点, 使用掩码创建一个新的数组，不包含最大的 n 个值
        centers = np.array(centers)
        mean_center = np.mean(centers, axis=0)
        distances = np.linalg.norm(np.squeeze(centers - mean_center), axis=1)
        indices_to_remove = np.argpartition(distances, -self.remove_div_points)[-self.remove_div_points:]
        mask = np.ones(centers.shape[0], dtype=bool)
        mask[indices_to_remove] = False
        filtered_centers = centers[mask]
        # 计算 去除散点的中心
        centers = np.array(filtered_centers)
        mean_center = np.mean(centers, axis=0)
        distances = (np.linalg.norm(np.squeeze(centers - mean_center), axis=1))
        # 计算半径
        Rs = (np.linalg.norm(np.squeeze(contour - mean_center), axis=1))
        R = np.mean(Rs)
        return (np.mean(distances) < self.threshold ) , mean_center ,R
    