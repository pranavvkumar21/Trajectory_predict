#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import utils
import os
from kalman_predict import *
cap = cv.VideoCapture("test.mp4")
size= 480
kernel = np.ones((3,3),np.uint8)
dilate_kernel = np.ones((20,20),np.uint8)
ok = False
scale = 60
fps = 240
dt = 1/fps
state_size = 4   #cx,cy,vx,vy
measurement_size = 2  #cx,cy
control_size = 2
noise = 3

sigmaM = 1e-4
sigmaZ = 3*noise

flag=0
vx = [-500,-500]
pos = np.empty((0,2))

mu = np.array([30.0,30.0,0.0,0.0]).reshape((state_size,1))
P = np.diag([1000.0,1000.0,1000.0,1000.0])**2
acc = np.array([0.0,900.0]).reshape((control_size,1))

def color_track(frame):
    img = frame.copy()
    frame = utils.g_blur(frame,5,2)
    mask = utils.detect_ball(frame)
    res = cv.bitwise_and(frame,frame, mask= mask)
    mask = cv.dilate(mask,dilate_kernel,iterations = 4)
    res = cv.dilate(res,dilate_kernel,iterations = 4)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        cnt = contours[0]
        bbox = cv.boundingRect(cnt)
        bbox_c = bbox
        centre_c = (bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2)
        #cv.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),2)
        cv.circle(img,centre_c,10,(0,0,255),5)
        cv.putText(img, "ball_c", (bbox[0],bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
        return img,bbox_c,centre_c
def object_track(img,tracker):
    ok,bbox = tracker.update(img)
    centre_o = (bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2)
    cv.circle(img,centre_o,10,(0,255,0),5)
    cv.putText(img, "ball", (centre_o[0]+5,centre_o[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
    return img


while cap.isOpened():

    _, frame = cap.read()
    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)
    dim = (width, height)
    frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    if True:
        dt = 1/240
        img,bbox_c,c = color_track(frame)
        kalman.predict(acc,1)
        mu,sig = kalman.correct(np.array([c[0],c[1]]).reshape((2,1)))
        for i in range(100):
            mu,sig,zp = kalman.predict(acc,)
            #print(str(int(mu[0]))+"\t"+str(int(mu[1])))
            #print(img.shape)
            cv.circle(img,(int(mu[0]),int(mu[1])),10,(0,255,0),5)

    if flag:
        img = object_track(img,tracker)


    cv.imshow('frame',img)


    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    if k ==ord("a"):
        print(bbox_c)
        tracker = utils.get_tracker("csrt")
        flag = tracker.init(img,bbox_c)
        flag=True
        ok = True
cv.destroyAllWindows()
