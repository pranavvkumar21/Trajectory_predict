#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import numpy.linalg as la

fps = 240
dt = 1/fps
state_size = 4   #cx,cy,vx,vy
measurement_size = 2  #cx,cy
control_size = 2
noise = 3

sigmaM = 0.3
sigmaZ = 0.3*noise


class KalmanFilter:
    def __init__(self,F,H,B,Q,R):
        #state, measurement and control matrices
        self.F = F
        self.H = H
        self.B = B
        #Prediction noise and measurement noises
        self.Q = Q
        self.R = R
        #Initial mean and covariance

    def predict(self,u,mu,cov):
        mu = np.matmul(self.F,mu) + np.matmul(self.B,u)
        cov = np.matmul(np.matmul(self.F,cov),self.F.T)+self.Q
        z_p = np.matmul(self.H,mu)
        return mu,cov,z_p
    def correct(self,z,mu,cov,z_p):
        eps = z-z_p
        K = np.matmul(np.matmul(cov,self.H.T),la.inv(np.matmul(np.matmul(self.H,cov),self.H.T)+self.R))
        mu += np.matmul(K,eps)
        cov = np.matmul((np.eye(len(cov))-np.matmul(K,self.H)),cov)
        return mu,cov




mu = np.array([0.0,0.0,0.0,0.0]).reshape((state_size,1))
cov = np.diag([1000.0,1000.0,1000.0,1000.0])**2
acc = np.array([0.0,900.0]).reshape((control_size,1))

#kalman = KalmanFilter(state_size,measurement_size,control_size)
transitionMatrix = np.array(                        #F matix
[1, 0, dt, 0,
 0, 1, 0, dt,
 0, 0, 1, 0,
 0, 0, 0, 1]
).reshape((state_size,state_size))

measurementMatrix = np.array([                 #H matrix
1,0,0,0,
0,1,0,0,
]).reshape((measurement_size,state_size))

controlMatrix = np.array([                     #B matrix
dt**2, 0,
0, dt**2,
dt, 0,
0, dt
]).reshape((state_size,control_size))

M_noise = sigmaM**2 * np.eye(4)         #Q Matrix
Z_noise = sigmaZ**2 * np.eye(2)     #R matrix
kalman = KalmanFilter(transitionMatrix,measurementMatrix,controlMatrix,M_noise,Z_noise)

#for i in range(3):
#prediction = kalman.predict(a)
#    print(prediction[0])
