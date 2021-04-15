# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
# 顺时针旋转图片90
def rotate(img, point):
    h, w = img.shape[:2]
    img = np.transpose(img, (1, 0, 2))
    img = img[:, ::-1]
    x, y = point
    return img.copy(), (h - y, x)


# 逆时针旋转图片90
def rotate_(img, point):
    h, w = img.shape[:2]
    img = np.transpose(img, (1, 0, 2))
    img = img[::-1]
    x, y = point
    return img.copy(), (y, w - x)
if __name__ == "__main__":
    pass