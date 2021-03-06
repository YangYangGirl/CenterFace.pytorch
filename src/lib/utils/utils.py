from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import random
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


def Data_body_anchor_sample(image, anns):
    maxSize = 12000
    infDistance = 9999999

    boxes = []
    for ann in anns:
        if ann['lefthand_valid'] is True:
            boxes.append([ann['lefthand_box'][0], ann['lefthand_box'][1], ann['lefthand_box'][0]+ann['lefthand_box'][2], ann['lefthand_box'][1]+ann['lefthand_box'][3]])
        if ann['righthand_valid'] is True:
            boxes.append([ann['righthand_box'][0], ann['righthand_box'][1], ann['righthand_box'][0]+ann['righthand_box'][2], ann['righthand_box'][1]+ann['righthand_box'][3]])
    boxes = np.asarray(boxes, dtype=np.float32)
    height, width, _ = image.shape

    random_counter = 0

    boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    rand_idx = random.randint(0, len(boxArea)-1)
    rand_Side = boxArea[rand_idx] ** 0.5

    # anchors = [16, 32, 48, 64, 96, 128, 256, 512]
    anchors = [1, 2, 5, 13, 16, 20, 30, 42, 55, 62, 90, 105, 150] #2, 4, 16, 20, 24, 28, 32, 48, 64, 80, 96, 128] 
    distance = infDistance
    anchor_idx = 5
    for i, anchor in enumerate(anchors):
        if abs(anchor - rand_Side) < distance:
            distance = abs(anchor - rand_Side)  # 选择最接近的anchors
            anchor_idx = i

    target_anchor = random.choice(anchors[0:min(anchor_idx+1, 6) ])  # 随机选择一个相对较小的anchor，向下
    ratio = float(target_anchor) / rand_Side  # 缩放的尺度

    ratio = ratio * (2 ** random.uniform(-1, 1))  # [ratio/2, 2ratio]的均匀分布

    if int(height * ratio * width * ratio) > maxSize * maxSize:
        ratio = (maxSize * maxSize / (height * width)) ** 0.5

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)

    ratio = np.random.choice(np.arange(0.9, 2, 0.1))
    image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)

    boxes[:, 0] *= ratio
    boxes[:, 1] *= ratio
    boxes[:, 2] *= ratio
    boxes[:, 3] *= ratio

    boxes = boxes.tolist()
    which_bbox = 0
    for i in range(len(anns)):
        if anns[i]['lefthand_valid'] is True:
            anns[i]['lefthand_box'] = [boxes[which_bbox][0], boxes[which_bbox][1], boxes[which_bbox][2]-boxes[which_bbox][0], boxes[which_bbox][3]-boxes[which_bbox][1]]      # 左手bbox
            for j in range(21):
                anns[i]['lefthand_kpts'][j*3] *= ratio
                anns[i]['lefthand_kpts'][j*3+1] *= ratio
            which_bbox += 1
        if anns[i]['righthand_valid'] is True:
            anns[i]['righthand_box'] = [boxes[which_bbox][0], boxes[which_bbox][1], boxes[which_bbox][2]-boxes[which_bbox][0], boxes[which_bbox][3]-boxes[which_bbox][1]]      # 右手bbox
            for j in range(21):
                anns[i]['righthand_kpts'][j*3] *= ratio
                anns[i]['righthand_kpts'][j*3+1] *= ratio
            which_bbox += 1

    return image, anns


def Data_anchor_sample(image, anns):
    maxSize = 12000
    infDistance = 9999999

    boxes = []
    for ann in anns:
        boxes.append([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]])
    boxes = np.asarray(boxes, dtype=np.float32)

    height, width, _ = image.shape

    random_counter = 0

    boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    rand_idx = random.randint(0, len(boxArea)-1)
    rand_Side = boxArea[rand_idx] ** 0.5

    anchors = [16, 32, 48, 64, 96, 128, 256, 512]
    
    distance = infDistance
    anchor_idx = 5
    for i, anchor in enumerate(anchors):
        if abs(anchor - rand_Side) < distance:
            distance = abs(anchor - rand_Side)  # 选择最接近的anchors
            anchor_idx = i

    target_anchor = random.choice(anchors[0:min(anchor_idx+1, 5) ])  # 随机选择一个相对较小的anchor，向下
    ratio = float(target_anchor) / rand_Side  # 缩放的尺度
    ratio = ratio * (2 ** random.uniform(-1, 1))  # [ratio/2, 2ratio]的均匀分布

    if int(height * ratio * width * ratio) > maxSize * maxSize:
        ratio = (maxSize * maxSize / (height * width)) ** 0.5

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)
    image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)

    boxes[:, 0] *= ratio
    boxes[:, 1] *= ratio
    boxes[:, 2] *= ratio
    boxes[:, 3] *= ratio

    boxes = boxes.tolist()
    for i in range(len(anns)):
        anns[i]['bbox'] = [boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]]      # 人脸bbox
        for j in range(5):
            anns[i]['keypoints'][j*3] *= ratio
            anns[i]['keypoints'][j*3+1] *= ratio

    return image, anns