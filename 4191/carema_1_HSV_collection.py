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

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(3, 3, 1, 0.8, aruco_dict)
length_of_axis = 0.01



if __name__ == '__main__':

    
    vid = cv2.VideoCapture(2)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 1280
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 720
    #vid.set(cv2.CAP_PROP_FPS, 40)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    color_list = []
    data = 0
    data_out = np.zeros((1, 3))

    while True:
        #print('------------START NEW ITER-------------')
        ts = time.time()
        t0 = time.time()
        ret, img = vid.read()
        key = cv2.waitKey(1) & 0xFF
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        x_l = 335
        y_l = 150
        x_r = 350
        y_r = 170
        cv2.rectangle(img, (x_l, y_l), (x_r, y_r), (0, 255, 0), 2)

        cut_part = img_hsv[y_l:y_r,x_l:x_r,:]

        '''
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        if key == ord('e'):
            mean_color = cv2.mean(cut_part)
            temp = np.round(mean_color)[0:3]
            print(np.round(mean_color))
            print(temp)
            data_out = np.append(data_out, [temp], axis=0)
            file_name = 'output.out'
            np.savetxt(file_name, data_out)
        

        #mean_color = cv2.mean(img_hsv)
        cv2.imshow('img',img)
        #print('time used all', time.time() - ts)
        time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
