# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import json

if __name__ == "__main__":
    np.random.seed(50)
    img_dir = r'D:\datasets\mpii\mpii_human_pose_v1\images'
    with open('mpii_train.json', 'r') as f:
        data = json.load(f)
    names = list(data.keys())
    np.random.shuffle(names)
    for name in names[:10]:

        anns = data[name]
        print(anns)
        img = cv2.imread(os.path.join(img_dir, name))
        for ann in anns:
            bbox = ann['bbox']
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            hh = ann['scale'] * 200
            x, y = ann['center']
            xx1, xx2 = x - hh / 2, x + hh / 2
            yy1, yy2 = y - hh / 2, y + hh / 2
            xx1, yy1, xx2, yy2 = int(xx1), int(yy1), int(xx2), int(yy2)
            cv2.rectangle(img, (xx1, yy1), (xx2, yy2), (0, 0, 255), 2)
            points = ann['points']
            for point in points:
                x, y, p_id, vi = point
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 2, (0, 255, 255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(1000)
