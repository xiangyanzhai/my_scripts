# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import cv2 as cv
import time
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)])
path = r'C:\Users\Sunny\Desktop\sxq'

path_orange = r'C:\Users\Sunny\Desktop\sxq\orange'
path_blue = r'C:\Users\Sunny\Desktop\sxq\blue'
path_black = r'C:\Users\Sunny\Desktop\sxq\black'
ball_num_max = 5
shadow_num_max = 5
Img = []
for name in os.listdir(path):
    if 'png' not in name:
        continue
    file = os.path.join(path, name)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    Img.append(img)


def get_img(path):
    R = []
    names = os.listdir(path)
    names = sorted(names)
    print(names)
    l = len(names) // 2
    for i in range(l):
        r = []
        file = os.path.join(path, names[i * 2])
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        r.append(img)
        file = os.path.join(path, names[i * 2 + 1])
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        r.append(img)
        R.append(r)
    return R


def img_transform(img):
    alpha_img = img[..., -1]
    img = img[..., :-1]
    img = img[..., ::-1]

    img = Image.fromarray(img)
    img = transform(img)
    img = np.array(img)
    img = img[..., ::-1]
    alpha_img = alpha_img[..., None]
    img = np.concatenate([img, alpha_img], axis=-1)

    return img


Img_num = len(Img)

Img_orange = get_img(path_orange)
Img_blue = get_img(path_blue)
Img_black = get_img(path_black)
print(len(Img_orange), len(Img_blue), len(Img_black))


def img_scale(img, img_min=20, img_max=50):
    h, w = img.shape[:2]
    scale1 = img_min / min(h, w)
    scale2 = img_max / max(h, w)
    scale = np.random.uniform(scale1, scale2)
    return scale, np.ceil(h * scale), np.ceil(w * scale)


def motion_blur(image, degree=12, angle=45):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def paste_ROI_to_image(image, ROI, paste_area):
    # image = image_in.copy()
    y1, x1, y2, x2 = paste_area

    ROI = cv.resize(ROI, (x2 - x1 + 1, y2 - y1 + 1))  # cv.resize(src, dsize=(width, height))

    image = image.astype(np.float)
    ROI = ROI.astype(np.float)

    # alpha通道
    alpha_image = image[y1:y2 + 1, x1:x2 + 1, 3] / 255.0
    alpha_ROI = ROI[:, :, 3] / 255.0
    alpha = 1 - (1 - alpha_image) * (1 - alpha_ROI)
    # BGR通道
    for i in range(3):
        image[y1:y2 + 1, x1:x2 + 1, i] = (image[y1:y2 + 1, x1:x2 + 1, i] * alpha_image * (1 - alpha_ROI) + ROI[:, :,
                                                                                                           i] * alpha_ROI) / alpha

    image[y1:y2 + 1, x1:x2 + 1, 3] = alpha * 255
    image = image.astype(np.uint8)

    return image


def generate_shadow(bg_img):
    edge_length = np.random.randint(22, 72)
    patch = np.zeros((edge_length, edge_length, 3), dtype=np.uint8)
    alpha_img = np.zeros((edge_length, edge_length), dtype=np.uint8)
    angle = np.random.randint(-45, 45)
    center = int(edge_length / 2), int(edge_length / 2)
    a = np.random.randint(5, int(edge_length / 2 - 1))
    b = np.random.randint(5, int(edge_length / 2 - 1))
    # color = np.random.randint(10, 26) * 10 + np.random.randint(10)
    # color = min(color, 255)
    color = np.random.randint(80, 150)
    cv2.ellipse(alpha_img, center, (a, b), angle, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(alpha_img, center, (a, b), angle, 0, 360, color, -1, 3)
    patch = np.concatenate([patch, alpha_img[..., None]], axis=-1)
    if np.random.random() > 0.5:
        patch = motion_blur(patch, degree=12, angle=45)

    h, w = bg_img.shape[:2]

    y1, x1 = int(h / 2) + np.random.randint(int(h / 2) - edge_length - 1), np.random.randint(w - edge_length - 1)

    y2, x2 = y1 + edge_length, x1 + edge_length
    bg_img = paste_ROI_to_image(bg_img, patch, [y1, x1, y2 - 1, x2 - 1])

    return bg_img


def rand_img():
    rand = np.random.random()

    if rand < 0.6:
        index = np.random.choice(len(Img_orange))
        img = Img_orange[index]
    elif rand < 0.9:
        index = np.random.choice(len(Img_blue))
        img = Img_blue[index]
    else:
        index = np.random.choice(len(Img_black))
        img = Img_black[index]

    rand = np.random.random()
    if rand < 5 / 6:

        img = img[0]
        if np.random.random() > 0.5:
            img = img[::-1]
    else:
        img = img[1]
    return img


def fill_img(bg_img):
    bboxes = []
    bg_img_h, bg_img_w = bg_img.shape[:2]
    num = np.random.randint(ball_num_max) + 1
    img_index = np.random.choice(Img_num, num, replace=True)

    Scale = []
    HW = []
    Current_img = []
    for i in range(num):
        img = rand_img()
        Current_img.append(img)
        scale, h, w = img_scale(img)
        Scale.append(scale)
        HW.append([h, w])

    HW = np.array(HW)
    HW = HW.max(axis=0)
    h, w = HW
    rows = int(bg_img_h / (h * 2))
    cols = int(bg_img_w / (w * 2))
    loc_num = rows * cols
    num = min(loc_num, num)
    loc = list(range(loc_num))
    np.random.shuffle(loc)

    alpha_image = np.ones(bg_img.shape[:2], dtype=np.uint8)[..., None] * 255

    bg_img = np.concatenate([bg_img, alpha_image], axis=-1)
    for i in range(np.random.randint(shadow_num_max)):
        bg_img = generate_shadow(bg_img)

    for i in range(num):
        row = loc[i] // cols
        col = loc[i] % cols
        y1, x1 = row * h * 2 + np.random.randint(h), col * w * 2 + np.random.randint(w)
        img = Current_img[i]
        img = img_transform(img)
        img_h, img_w = img.shape[:2]
        factor = np.random.randint(6, 11) / 10
        n_h, n_w = int(img_h * factor * Scale[i]), int(img_w * factor * Scale[i])

        if np.random.random() > 0.5:
            img = img[:, ::-1]
        if np.random.random() > 0.5:
            img = motion_blur(img, degree=12, angle=45)

        y1, x1 = int(y1), int(x1)
        y2, x2 = y1 + n_h, x1 + n_w

        bg_img = paste_ROI_to_image(bg_img, img, [y1, x1, y2 - 1, x2 - 1])

        bboxes.append([y1, x1, y2, x2])

    bboxes = np.array(bboxes, dtype=np.float32)
    bboxes[:, slice(0, 4, 2)] = np.clip(bboxes[:, slice(0, 4, 2)], 0, bg_img_h - 1)
    bboxes[:, slice(1, 4, 2)] = np.clip(bboxes[:, slice(1, 4, 2)], 0, bg_img_w - 1)
    return bg_img, bboxes


if __name__ == "__main__":
    pass
    import joblib
    import codecs

    path = r'G:\IMG3\a'
    path = r'H:\datasets\val2017\val2017'
    # writedir = r'H:\datasets\val2017_solid'
    # path = r'H:\datasets\playground'
    # path = r'G:\IMG3\a'
    # writedir = r'H:\datasets\playground_solid'
    # writedir = r'H:\datasets\playground_gen'
    writedir = r'H:\datasets\val2017_gen_2'

    path=r'H:\datasets\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
    writedir=r'H:\datasets\VOCtest2007_gen'
    writedir = r'H:\datasets\VOCtest2007_gen_2'
    names = os.listdir(path)
    Res = {}
    for name in names[:]:
        try:

            file = os.path.join(path, name)

            bg_img = cv2.imread(file)
            h, w = bg_img.shape[:2]

        except:
            continue

        if h <= 200 or w <= 200:
            continue
        s = time.time()
        bg_img, bboxes = fill_img(bg_img)
        print(time.time() - s)

        n_bboxes = []
        with codecs.open(os.path.join(writedir, name.split('.')[0] + '.txt'), 'w') as f:
            for bbox in bboxes:
                y1, x1, y2, x2 = bbox
                # cv2.rectangle(bg_img, (x1, y1), (x2, y2), (0, 255, 255))
                n_bboxes.append([x1, y1, x2, y2, 0, 0, 0])

                y1,x1,y2,x2=bbox
                x,y,ww,hh=(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1
                x=x/w
                y=y/h
                ww=ww/w
                hh=hh/h

                s = '0' + ' ' + str(round(x,7)) + ' ' + str(round(y,7)) + ' ' + str(round(ww,7)) + ' ' + str(round(hh,7)) + '\n'
                #         f.write(s)
                # s = '0' + ' ' + str(int(y1)) + ' ' + str(int(x1)) + ' ' + str(int(y2)) + ' ' + str(int(x2)) + '\n'
                f.write(s)


        n_bboxes = np.array(n_bboxes)

        Res[name.split('.')[0]] = n_bboxes
        cv2.imwrite(os.path.join(writedir, name), bg_img)
        # cv2.imshow('img', bg_img)
        # cv2.waitKey(2000)
joblib.dump(Res, 'VOCtest2007_gen_2.pkl')

# with codecs.open(os.path.join(writedir, name.split('.')[0] + '.txt'), 'w') as f:
#     yx = (bboxes[:, :2] + bboxes[:, 2:4]) / 2
#     hw = bboxes[:, 2:4] - bboxes[:, :2] + 1
#     bboxes = np.concatenate([yx, hw], axis=-1)
#     bboxes = bboxes[:, [1, 0, 3, 2]]
#     bboxes = bboxes / np.array([w, h, w, h])
#     bboxes = np.round(bboxes, 6)
#     for bbox in bboxes:
#         x, y, w, h = bbox
#
#         s = '0' + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
#         f.write(s)
#
#     # cv2.rectangle(bg_img, (x1, y1), (x2, y2), (0, 255, 255))
#
# cv2.imwrite(os.path.join(writedir, name), bg_img)
# cv2.imshow('img', bg_img)
# cv2.waitKey(2000)
# joblib.dump(Res, 'playground_solid_gt.pkl')
