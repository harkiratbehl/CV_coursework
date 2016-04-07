#!/usr/bin/python

import cv2
from sklearn.linear_model import LinearRegression

S = cv2.imread("img_color.jpg") # Source color image
T = cv2.imread("img_gray.jpg") # Target gray scale image

