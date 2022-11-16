#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import os

folder = "./dataset_arrays/"
kernel = np.ones((3,3),np.uint8)
dilate_kernel = np.ones((20,20),np.uint8)

def load_data():
    os.chdir(folder)
    arrs = os.listdir()
    X = np.empty((0,4,2))
    Y = np.empty((0,1,2))
    for i in arrs:
        train = np.load(i)
        x = train[:,:-1,:]
        y = train[:,-1,:].reshape((-1,1,2))
        X = np.concatenate([X,x])
        Y = np.concatenate([Y,y])
    print(X.shape)
    print(Y.shape)
    os.chdir("..")
    return X,Y

def resize(frame,size):
    max_shape = min(frame.shape[0],frame.shape[1])//2
    cent_x = frame.shape[1]//2
    cent_y = frame.shape[0]//2
    frame = frame[cent_y-max_shape:cent_y+max_shape,cent_x-max_shape:cent_x+max_shape,:]
    frame = cv.resize(frame,(size,size), interpolation = cv.INTER_AREA)
    return frame
def g_blur(img,k_size,iterations):
    for i in range(iterations):
        frame = cv.GaussianBlur(img,(k_size,k_size),0)
    return frame
def detect_ball(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    col = np.uint8([[[0,255,142 ]]])
    col = cv.cvtColor(col, cv.COLOR_BGR2HSV)[0,0]
    lower = np.array([col[0]-20,120,100])
    upper = np.array([col[0]+10,255,255])

    mask = cv.inRange(hsv, lower, upper)
    mask = cv.GaussianBlur(mask,(5,5),0)

    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel,iterations=3)
    return mask

def list_split(cou,length):
    if len(cou)<length:
        return []
    elif len(cou)==length:
        return cou
    else:
        sublist = []
        for i in range(0,len(cou)-length):
            sub = cou[i:i+length]
            sublist.append(sub)
        return sublist
