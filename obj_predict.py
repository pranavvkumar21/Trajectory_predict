#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import utils
import os
cap = cv.VideoCapture(0)
size= 480
kernel = np.ones((3,3),np.uint8)
dilate_kernel = np.ones((20,20),np.uint8)
ok = False
split_size=5
frame_size = 256
ratio = 256//480
buff = np.empty((0,split_size,2))
flag=0

for i in range(10000):
    # Take each frame

    _, frame = cap.read()
    img = frame.copy()
    if True:
        frame = utils.g_blur(frame,5,2)
        mask = utils.detect_ball(frame)
        res = cv.bitwise_and(frame,frame, mask= mask)
        mask = cv.dilate(mask,dilate_kernel,iterations = 4)
        res = cv.dilate(res,dilate_kernel,iterations = 4)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            cnt = contours[0]
            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            bbox = cv.boundingRect(cnt)
            bbox_c = bbox
            centre_c = (bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2)
            #cv.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),2)
            cv.circle(img,centre_c,10,(0,0,255),5)
            cv.putText(img, "ball_c", (bbox[0],bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
    if flag:
        ok,bbox = tracker.update(img)
        #cv.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),2)
        centre_o = (bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2)
        cv.circle(img,centre_o,10,(0,255,0),5)

        cv.putText(img, "ball", (centre_o[0]+5,centre_o[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
    cv.imshow('frame',img)


    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    if k ==ord("a"):
        print(bbox)
        tracker = utils.get_tracker("csrt")
        flag = tracker.init(img,bbox_c)
        flag=True
        ok = True
cv.destroyAllWindows()
