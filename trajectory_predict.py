#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import utils
import os
cap = cv.VideoCapture(0)
size= 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
kernel = np.ones((3,3),np.uint8)
dilate_kernel = np.ones((20,20),np.uint8)
prev = [float("inf"),float("inf")]
mov=1000
count=0
start_img=0
stop_img=0
cou=[]
split_size=5
frame_size = 256
ratio = 256//480
buff = np.empty((0,split_size,2))
def dist(prev,curr):
    return ((prev[0]-curr[0])**2+(prev[1]-curr[1])**2)/2

for i in range(10000):
    # Take each frame
    flag=0
    _, frame = cap.read()
    #print(frame.shape)
    #frame = utils.resize(frame,frame_size)
    #print(frame.shape)
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

        cv.line(frame,(cx-10,cy),(cx+10,cy),(255,0,0),5)
        cv.line(frame,(cx,cy-10),(cx,cy+10),(255,0,0),5)
        curr = [cx,cy]
        if prev[0]!=float("inf"):
            cv.line(frame,(cx,cy),(prev[0],prev[1]),(0,0,255),5)
        dis = dist(prev,curr)
        #print(dis)
        prev=curr
        prev_mov = mov
        mov = 0 if dis<150 else 1
        if mov==1 and prev_mov==0:
            count+=1
            start_img = frame

        if mov ==0 and prev_mov==1:
            for pos in range(len(cou)-1):
                cv.line(frame,(cou[pos][0],cou[pos][1]),(cou[pos+1][0],cou[pos+1][1]),(0,0,255),5)
                cv.circle(frame,(cou[pos][0],cou[pos][1]),10,(0,255,0),5)
            stop_img = frame
            main_list = cou.copy()
            flag=1


            cou=[]
        if mov:
            cou.append((curr))

    cv.imshow('frame',frame)


    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    if flag and len(main_list)>split_size :
        #print(len(main_list))
        a = input("save ?: ")
        if a=="y":
            sublist = utils.list_split(main_list,split_size)
            sample = np.array(sublist)
            print(sample.shape)
            buff = np.concatenate([buff,sample])
            print(buff.shape)
        if a=="s":
            l = len(os.listdir("./dataset_arrays/"))+1
            np.save("./dataset_arrays/array_"+str(l),buff)
            print("saved")
            buff = np.empty((0,split_size,2))
cv.destroyAllWindows()
