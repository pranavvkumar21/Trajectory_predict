# Tennis Ball Trajectory Prediction using Computer Vision and Kalman Filter

In this project, we use computer vision and Kalman filter to predict the trajectory of a tennis ball. The code is written in Python and uses the OpenCV and NumPy libraries.

## Overview

The model uses a video file or camera feed as input and detects the ball to be tracked using color-based segmentation. Color tracking was used instead of object tracking as it performed better when working with a webcam. The code then uses a Kalman filter to predict the position of the ball in the next frame. The predicted position is then used to update the Kalman filter for the next iteration. The predicted trajectory of the ball is displayed in real-time on the video feed.

## Usage

To predict the trajectory of a tennis ball using computer vision and Kalman filter, follow these steps:
1. change the cap_file variable to the desired vide file path or set to 0 to use default camera.
2. Run obj_predict.py
