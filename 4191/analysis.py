import cv2
import numpy as np
import time
from tennis_detect import tennis_detctor
    
if __name__ == '__main__':

    #vid = cv2.VideoCapture(2)
    #vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 1280
    #vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 720

    video_path = 'tennis/1.mp4'
    vid = cv2.VideoCapture(video_path)
    
    Tennis_Detector = tennis_detctor(length_filter = 30,
                                       remove_div_points = 8,
                                       random_points = 20,
                                       threshold = 4) 
    
    while True:

        ret, frame = vid.read()
        if not ret:
            vid = cv2.VideoCapture(video_path)
            ret, frame = vid.read()  
            
            
        frame = cv2.imread('tennis/11.jpg')
        
        Original = frame.copy()
        cv2.imshow('Original',Original)
        
        ts = time.time() # record time usage
        C,R = Tennis_Detector.detect(frame)
        #print('time used all', time.time() - ts)

        for i in range(len(R)):
            R_int = R[i]
            x,y = C[i]

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
            cv2.circle(frame, (x,y), 1, (0,0,255), 8)
            cv2.rectangle(frame, (x-R_int, y-R_int), (x + R_int, y + R_int), (0, 0, 255), 3)
            print('circle number : ', i+1 , 'position ' , x-320, y-240)
        
        x_offset = 800
        y_offset = 600
        # 显示结果图像
        cv2.imshow('tennis',frame)
        #cv2.moveWindow('tennis', x_offset * 1, y_offset * 2)
        cv2.imshow('Original',Original)
        #cv2.moveWindow('Original', x_offset * 2-150, y_offset * 2)
        
        time.sleep(0.02)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

