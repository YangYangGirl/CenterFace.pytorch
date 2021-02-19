from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data
import sys
sys.path.append("../")
from experiments.wholebody_evaluation.evaluation_wholebody import eval_lefthand_datasets


class WHOLEBODY(data.Dataset):
  num_classes = 1
  num_joints = 21
  
  # default_resolution = [800, 800]
  default_resolution = [576, 576]
  # default_resolution = [384, 384]
  # default_resolution = [256, 256]
  # default_resolution = [224, 224]
  
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  flip_idx = []
  #flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
  def __init__(self, opt, split):
    super(WHOLEBODY, self).__init__()
    self.anns_id = -1
    self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                  [4, 6], [3, 5], [5, 6], 
                  [5, 7], [7, 9], [6, 8], [8, 10], 
                  [6, 12], [5, 11], [11, 12], 
                  [12, 14], [14, 16], [11, 13], [13, 15]]
    
    self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    self.data_dir = os.path.join(opt.data_dir, 'coco')
    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'coco_wholebody_{}_v1.0.json').format(split)
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations', 
        'coco_wholebody_{}_v1.0.json').format(split)
    self.max_objs = 32
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt
    if self.num_classes == 1:
      self.class_name = ['__background__', 'lefthand']
      self._valid_ids = [1, 2]
    elif self.num_classes == 2:
      self.class_name = ['__background__', 'lefthand', 'righthand']
      self._valid_ids = [1, 2, 3]
    
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

    print('==> initializing coco wholebody {} data.'.format(split))
    print('==> load annotations from', self.annot_path)
    self.coco = coco.COCO(self.annot_path)
    
    image_ids = self.coco.getImgIds()

    if split == 'train':
      self.images = []
      for img_id in image_ids:
        flag = 0
        idxs = self.coco.getAnnIds(imgIds=[img_id])
        if len(idxs) > 0:
          for ench_ann_ids in idxs:
            anns = self.coco.loadAnns(ids=ench_ann_ids)
            for a in anns:
              if a['lefthand_valid'] is True or a['righthand_valid'] is True:
                flag = 1
        if flag == 1:
          self.images.append(img_id)
    else:
      self.images = []
      for img_id in image_ids:
        flag = 0
        idxs = self.coco.getAnnIds(imgIds=[img_id])
        if len(idxs) > 0:
          for ench_ann_ids in idxs:
            anns = self.coco.loadAnns(ids=ench_ann_ids)
            for a in anns:
              if a['lefthand_valid'] is True or a['righthand_valid'] is True:
                flag = 1
        if flag == 1:
          self.images.append(img_id)
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # results_json = {}
    # results_json["info"] = {
    #     "description": "COCO-WholeBody",
    #     "url": "https://github.com/jin-s13/COCO-WholeBody",
    #     "version": "1.0",
    #     "year": "2020",
    #     "date_created": "2020/07/01"
    # }

    # results_json["licenses"] = {}
    # results_json["categories"] = [
    #   {
    #     "supercategory": "person",
    #     "id": 1,
    #     "name": "person",
    #     "keypoints": [
    #         "nose",
    #         "left_eye",
    #         "right_eye",
    #         "left_ear",
    #         "right_ear",
    #         "left_shoulder",
    #         "right_shoulder",
    #         "left_elbow",
    #         "right_elbow",
    #         "left_wrist",
    #         "right_wrist",
    #         "left_hip",
    #         "right_hip",
    #         "left_knee",
    #         "right_knee",
    #         "left_ankle",
    #         "right_ankle"
    #     ],
    #     "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],\
    #       [9,11],[2,3],[1,2],[1,3], [2,4], [3,5],[4,6],[5,7]]
    #   }

    detections = []
    if self.num_classes == 2:
      for image_id in all_bboxes:
        for cls_ind in all_bboxes[image_id]:
          category_id = 1
          for dets in all_bboxes[image_id][cls_ind]:
            score = dets[4]

            if score > 0:
              self.anns_id += 1
              bbox = dets[:4]
              w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
              bbox[2] -= bbox[0]
              bbox[3] -= bbox[1]
            
              bbox_out  = list(map(self._to_float, bbox))

              if np.array(dets).shape[0] == 47:
                keypoints = np.concatenate([
                  np.array(dets[5:47], dtype=np.float32).reshape(-1, 2), 
                  np.ones((21, 1), dtype=np.float32)], axis=1).reshape(63).tolist()
                keypoints  = list(map(self._to_float, keypoints))

                detection = {
                    'id': self.anns_id, 
                    "image_id": int(image_id),
                    "category_id": self._valid_ids[cls_ind - 1], 
                    "score": float("{:.2f}".format(score)),
                    "area": w * h,
                    "scale": [0.0, 0.0],
                    "foot_kpts": [],
                    "face_kpts": [],
                    "center": [bbox_out[0] + w/2, bbox_out[1] + h/2]
                }

                if cls_ind == 1:
                  detection['lefthand_valid'] = True,
                  detection['leftthand_kpts'] = keypoints,
                  detection['lefthand_box'] = bbox_out,
                elif cls_ind == 2:
                  detection['righthand_valid'] = True,
                  detection['righthand_kpts'] = keypoints,
                  detection['righthand_box'] = bbox_out,

              else:
                detection = {
                    'id': self.anns_id, 
                    "image_id": int(image_id),
                    "category_id": 1, #self._valid_ids[cls_ind - 1], 
                    "score": float("{:.2f}".format(score)),
                    "area": w * h,
                    "scale": [0.0, 0.0],
                    "foot_kpts": [],
                    "face_kpts": [],
                    "center": [bbox_out[0] + w/2, bbox_out[1] + h/2],
                    "lefthand_valid": True,
                    "lefthand_kpts": [],
                    "lefthand_box": bbox_out,
                    "righthand_valid": True,
                    "righthand_kpts": [],
                    "righthand_box": bbox_out
                }
                # if cls_ind == 1:
                #   detection['lefthand_valid'] = True,
                #   detection['lefthand_kpts'] = [],
                #   detection['lefthand_box'] = bbox_out,
                #   detection['righthand_valid'] = False,
                #   detection['righthand_kpts'] = [],
                #   detection['righthand_box'] = [],
                # elif cls_ind == 2:
                #   detection['righthand_valid'] = True,
                #   detection['righthand_kpts'] = [],
                #   detection['righthand_box'] = bbox_out,
                #   detection['lefthand_valid'] = False,
                #   detection['lefthand_kpts'] = [],
                #   detection['lefthand_box'] = [],
              detections.append(detection)

    elif self.num_classes == 1:
      for image_id in all_bboxes:
        for cls_ind in all_bboxes[image_id]:
          category_id = 1
          for dets in all_bboxes[image_id][cls_ind]:
            score = dets[4]

            if score > 0.2:
              self.anns_id += 1
              bbox = dets[:4]
              w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
              bbox[2] -= bbox[0]
              bbox[3] -= bbox[1]
            
              bbox_out  = list(map(self._to_float, bbox))

              if np.array(dets).shape[0] == 47:
                keypoints = np.concatenate([
                  np.array(dets[5:47], dtype=np.float32).reshape(-1, 2), 
                  np.ones((21, 1), dtype=np.float32)], axis=1).reshape(63).tolist()
                keypoints  = list(map(self._to_float, keypoints))

                detection = {
                    'id': self.anns_id, 
                    "image_id": int(image_id),
                    "category_id": self._valid_ids[cls_ind - 1], 
                    "score": float("{:.2f}".format(score)),
                    "area": w * h,
                    "scale": [0.0, 0.0],
                    "foot_kpts": [],
                    "face_kpts": [],
                    "center": [bbox_out[0] + w/2, bbox_out[1] + h/2],
                    "lefthand_valid": True,
                    "lefthand_kpts": keypoints,
                    "lefthand_box": bbox_out,
                    "righthand_valid": True,
                    "righthand_kpts": keypoints,
                    "righthand_box": bbox_out
                }
              else:
                detection = {
                    'id': self.anns_id, 
                    "image_id": int(image_id),
                    "category_id": 1, #self._valid_ids[cls_ind - 1], 
                    "score": float("{:.2f}".format(score)),
                    "area": w * h,
                    "scale": [0.0, 0.0],
                    "foot_kpts": [],
                    "face_kpts": [],
                    "center": [bbox_out[0] + w/2, bbox_out[1] + h/2],
                    "lefthand_valid": True,
                    "lefthand_kpts": [],
                    "lefthand_box": bbox_out,
                    "righthand_valid": True,
                    "righthand_kpts": [],
                    "righthand_box": bbox_out
                }
              detections.append(detection)

    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results_0_2.json'.format(save_dir), 'w'))

  def run_eval(self, save_dir, results={}):
    # result_json = os.path.join(self.opt.save_dir, "results.json")
    # detections  = self.convert_eval_format(all_boxes)
    # json.dump(detections, open(result_json, "w"))
    if results != {}:
        self.save_results(results, save_dir)
        print("saving results to ", '{}/results_0_2.json'.format(save_dir))#, self.opt.exp_id))
    
    eval_lefthand_datasets('{}/results_0_2.json'.format(save_dir))

    # coco_dets = self.coco.loadRes('{}/results_0_2.json'.format(save_dir))
    # coco_eval = COCOeval(self.coco, coco_dets, "lefthand_kpts")
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    # coco_eval = COCOeval(self.coco, coco_dets, "lefthand_box")
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    


# class WHOLEBODY(data.Dataset):
#   num_classes = 2
#   num_joints = 21
  
#   default_resolution = [800, 800]
#   mean = np.array([0.40789654, 0.44719302, 0.47026115],
#                    dtype=np.float32).reshape(1, 1, 3)
#   std  = np.array([0.28863828, 0.27408164, 0.27809835],
#                    dtype=np.float32).reshape(1, 1, 3)
#   flip_idx = []
#   #flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
#   def __init__(self, opt, split):
#     super(WHOLEBODY, self).__init__()
#     self.anns_id = -1
#     self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
#                   [4, 6], [3, 5], [5, 6], 
#                   [5, 7], [7, 9], [6, 8], [8, 10], 
#                   [6, 12], [5, 11], [11, 12], 
#                   [12, 14], [14, 16], [11, 13], [13, 15]]
    
#     self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
#     self.data_dir = os.path.join(opt.data_dir, 'coco')
#     self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
#     if split == 'test':
#       self.annot_path = os.path.join(
#           self.data_dir, 'annotations', 
#           'coco_wholebody_{}_v1.0.json').format(split)
#     else:
#       self.annot_path = os.path.join(
#         self.data_dir, 'annotations', 
#         'coco_wholebody_{}_v1.0.json').format(split)
#     self.max_objs = 32
#     self._data_rng = np.random.RandomState(123)
#     self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
#                              dtype=np.float32)
#     self._eig_vec = np.array([
#         [-0.58752847, -0.69563484, 0.41340352],
#         [-0.5832747, 0.00994535, -0.81221408],
#         [-0.56089297, 0.71832671, 0.41158938]
#     ], dtype=np.float32)
#     self.split = split
#     self.opt = opt

#     print('==> initializing coco wholebody {} data.'.format(split))
#     print('==> load annotations from', self.annot_path)
#     self.coco = coco.COCO(self.annot_path)
    
#     image_ids = self.coco.getImgIds()

#     if split == 'train':
#       self.images = []
#       for img_id in image_ids:
#         flag = 0
#         idxs = self.coco.getAnnIds(imgIds=[img_id])
#         if len(idxs) > 0:
#           for ench_ann_ids in idxs:
#             anns = self.coco.loadAnns(ids=ench_ann_ids)
#             for a in anns:
#               if a['lefthand_valid'] is True or a['righthand_valid'] is True:
#                 flag = 1
#         if flag == 1:
#           self.images.append(img_id)
#     else:
#       self.images = []
#       for img_id in image_ids:
#         flag = 0
#         idxs = self.coco.getAnnIds(imgIds=[img_id])
#         if len(idxs) > 0:
#           for ench_ann_ids in idxs:
#             anns = self.coco.loadAnns(ids=ench_ann_ids)
#             for a in anns:
#               if a['lefthand_valid'] is True or a['righthand_valid'] is True:
#                 flag = 1
#         if flag == 1:
#           self.images.append(img_id)
#     self.num_samples = len(self.images)
#     print('Loaded {} {} samples'.format(split, self.num_samples))

#   def _to_float(self, x):
#     return float("{:.2f}".format(x))

#   def convert_eval_format(self, all_bboxes):
#     # results_json = {}
#     # results_json["info"] = {
#     #     "description": "COCO-WholeBody",
#     #     "url": "https://github.com/jin-s13/COCO-WholeBody",
#     #     "version": "1.0",
#     #     "year": "2020",
#     #     "date_created": "2020/07/01"
#     # }

#     # results_json["licenses"] = {}
#     # results_json["categories"] = [
#     #   {
#     #     "supercategory": "person",
#     #     "id": 1,
#     #     "name": "person",
#     #     "keypoints": [
#     #         "nose",
#     #         "left_eye",
#     #         "right_eye",
#     #         "left_ear",
#     #         "right_ear",
#     #         "left_shoulder",
#     #         "right_shoulder",
#     #         "left_elbow",
#     #         "right_elbow",
#     #         "left_wrist",
#     #         "right_wrist",
#     #         "left_hip",
#     #         "right_hip",
#     #         "left_knee",
#     #         "right_knee",
#     #         "left_ankle",
#     #         "right_ankle"
#     #     ],
#     #     "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],\
#     #       [9,11],[2,3],[1,2],[1,3], [2,4], [3,5],[4,6],[5,7]]
#     #   }

#     detections = []
#     for image_id in all_bboxes:
#       for cls_ind in all_bboxes[image_id]:
#         category_id = 1
#         for dets in all_bboxes[image_id][cls_ind]:
#           score = dets[4]

#           if score > 0.2:
#             self.anns_id += 1
#             bbox = dets[:4]
#             w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
#             bbox[2] -= bbox[0]
#             bbox[3] -= bbox[1]
          
#             bbox_out  = list(map(self._to_float, bbox))

#             if np.array(dets).shape[0] == 47:
#               keypoints = np.concatenate([
#                 np.array(dets[5:47], dtype=np.float32).reshape(-1, 2), 
#                 np.ones((21, 1), dtype=np.float32)], axis=1).reshape(63).tolist()
#               keypoints  = list(map(self._to_float, keypoints))

#               detection = {
#                   'id': self.anns_id, 
#                   "image_id": int(image_id),
#                   "category_id": int(category_id),
#                   "lefthand_box": bbox_out,
#                   "score": float("{:.2f}".format(score)),
#                   "lefthand_kpts": keypoints,
#                   "area": w * h,
#                   'lefthand_valid': True,
#                   "righthand_kpts": [],
#                   "scale": [0.0, 0.0],
#                   "foot_kpts": [],
#                   "face_kpts": [],
#                   "center": [bbox_out[0] + w/2, bbox_out[1] + h/2]
#               }
#             else:
#               detection = {
#                   'id': self.anns_id, 
#                   "image_id": int(image_id),
#                   "category_id": int(category_id),
#                   "lefthand_box": bbox_out,
#                   "score": float("{:.2f}".format(score)),
#                   "lefthand_kpts": [],
#                   "area": w * h,
#                   'lefthand_valid': True,
#                   "righthand_kpts": [],
#                   "scale": [0.0, 0.0],
#                   "foot_kpts": [],
#                   "face_kpts": [],
#                   "center": [bbox_out[0] + w/2, bbox_out[1] + h/2]
#               }

#             detections.append(detection)
#     return detections

#   def __len__(self):
#     return self.num_samples

#   def save_results(self, results, save_dir):
#     json.dump(self.convert_eval_format(results), 
#               open('{}/results_0_2.json'.format(save_dir), 'w'))

#   def run_eval(self, save_dir, results={}):
#     # result_json = os.path.join(self.opt.save_dir, "results.json")
#     # detections  = self.convert_eval_format(all_boxes)
#     # json.dump(detections, open(result_json, "w"))
#     if results != {}:
#         self.save_results(results, save_dir)
#         print("saving results to ", '{}/results_0_2.json'.format(save_dir))#, self.opt.exp_id))
    
#     eval_lefthand_datasets('{}/results_0_2.json'.format(save_dir))

#     # coco_dets = self.coco.loadRes('{}/results_0_2.json'.format(save_dir))
#     # coco_eval = COCOeval(self.coco, coco_dets, "lefthand_kpts")
#     # coco_eval.evaluate()
#     # coco_eval.accumulate()
#     # coco_eval.summarize()
#     # coco_eval = COCOeval(self.coco, coco_dets, "lefthand_box")
#     # coco_eval.evaluate()
#     # coco_eval.accumulate()
#     # coco_eval.summarize()
    


