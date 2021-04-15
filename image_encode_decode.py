# !/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np

img1=cv2.imread('1.jpg')
print(img1.shape)
x = cv2.imencode('.jpg', img1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
print(type(x))
img2 = cv2.imdecode(np.frombuffer(x,np.uint8), cv2.IMREAD_COLOR)
print(img2.shape)