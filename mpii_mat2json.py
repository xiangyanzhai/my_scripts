# !/usr/bin/python
# -*- coding:utf-8 -*-
import json
import scipy.io as scio

XX = 0
Y = 0
Z = 0


def process_single(x1, y1, x2, y2, scale, objpos, annopoints):
    # print(x1.shape, scale.shape, objpos.shape, annopoints.shape)
    x1 = x1[0][0]
    y1 = y1[0][0]
    x2 = x2[0][0]
    y2 = y2[0][0]
    scale = scale[0][0]
    objpos = objpos[0][0]
    x = objpos['x'][0][0]
    y = objpos['y'][0][0]
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    x, y = float(x), float(y)
    scale = float(scale)
    ann = {}
    ann['bbox'] = [x1, y1, x2, y2]
    ann['scale'] = scale
    ann['center'] = [x, y]
    points_list = []
    points = annopoints[0]['point'][0][0]
    for point in points:
        x, y, point_id, is_visible = point['x'], point['y'], point['id'], point['is_visible']
        x = x.ravel()[0]
        y = y.ravel()[0]
        point_id = point_id.ravel()[0]

        if point_id == 8 or point_id == 9:
            is_visible = 1
        else:
            is_visible = is_visible.ravel()[0]
        x = float(x)
        y = float(y)
        point_id = int(point_id)
        is_visible = int(is_visible)
        points_list.append([x, y, point_id, is_visible])

        # print(x, y, point_id, is_visible)

    ann['points'] = points_list

    return ann


def process(x1, y1, x2, y2, scale, objpos, annopoints):
    anns = []
    for i in range(x1.shape[0]):
        ann = process_single(x1[i], y1[i], x2[i], y2[i], scale[i], objpos[i], annopoints[i])
        anns.append(ann)
    return anns


import joblib

if __name__ == "__main__":
    file_path = r'D:\datasets\mpii\mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
    data = scio.loadmat(file_path)
    release = data['RELEASE']
    print(release.dtype)
    release = release[0][0]
    print(type(release), release.dtype)
    annolist = release['annolist']
    img_train = release['img_train']
    version = release['version']
    single_person = release['single_person']
    act = release['act']
    video_list = release['video_list']
    print('***********************************')
    print(annolist.shape, annolist.dtype)
    print(img_train.shape, img_train.dtype, img_train.sum())
    annolist = annolist[0]
    img_train = img_train[0]
    c = 0
    zz = 0
    data = {}
    for ann in annolist[:]:
        image = ann['image']
        annorect = ann['annorect']
        is_train = img_train[zz]
        zz += 1

        if is_train == 0 or 'annopoints' not in str(annorect.dtype) or 'x1' not in str(annorect.dtype):
            continue

        image = image.ravel()[0].ravel()[0][0][0]
        x1 = annorect['x1'][0]
        y1 = annorect['y1'][0]
        x2 = annorect['x2'][0]
        y2 = annorect['y2'][0]
        scale = annorect['scale'][0]
        objpos = annorect['objpos'][0]
        annopoints = annorect['annopoints'][0]
        # print(x1.shape, scale.shape, objpos.shape, annopoints.shape)
        anns = []
        # process(x1, y1, x2, y2, scale, objpos, annopoints)
        try:
            anns = process(x1, y1, x2, y2, scale, objpos, annopoints)
            XX += 1
        except Exception as e:
            print(e)
            Y += 1
            continue
            pass

        data[image] = anns

        # if x1.shape[0] > 1:
        #     break

        c += 1
    print('train_anns:', c, ',train_image:', XX, ',exception_image:', Y, ',total_anns:', zz)
    # joblib.dump(data, 'data.pkl')
    with open('mpii_train.json', 'w') as f:
        json.dump(data, f)
