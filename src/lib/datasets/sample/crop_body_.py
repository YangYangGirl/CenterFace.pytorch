from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.utils import Data_body_anchor_sample
from utils.Randaugmentations import Randaugment
import math
from PIL import Image
import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


class CropBodyDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    img = cv2.imread('../data/wider_face/crop_lefthand/imgs/' + img_id[:-1] + '.jpg')
    cls_id = 0
    with open('../data/wider_face/crop_lefthand/labels/' + img_id[:-1] + '.json','r') as load_f:
      pts_json = json.load(load_f)
      pts = np.array(pts_json['lefthand_keypoints'])
     
    origin_img = img.copy()
    result_img = img.copy()
    cv2.imwrite("origin_body_100.jpg", origin_img)

    for p in pts:
      result_img = cv2.circle(result_img, (int(p[0]), int(p[1])), 1, (0, 0, 255), 0)
    
    cv2.imwrite("origin_body_100_out.jpg", result_img)

    height, width = img.shape[0], img.shape[1]
    print("height, width",height, width)
    bbox = [0, 0, width, height]
    bbox = self._coco_box_to_bbox(bbox)
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        # s = s * np.random.choice(np.arange(0.8, 1.1, 0.1))
        s = s
        # _border = np.random.randint(128*0.4, 128*1.4)
        _border = s * np.random.choice([0.1, 0.2, 0.25])
        w_border = self._get_border(_border, img.shape[1])
        h_border = self._get_border(_border, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        yy =0
        # sf = self.opt.scale
        # cf = self.opt.shift
        # c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        # c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        # s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      # if np.random.random() < self.opt.aug_rot:
      #   rf = self.opt.rotate
      #   rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1

    # print("c, s, rot, [self.opt.input_res, self.opt.input_res]", c, s, rot, self.opt.input_res, self.opt.input_res)
    # trans_input = get_affine_transform(
    #   c, s, rot, [self.opt.input_res, self.opt.input_res])
    # print("trans_input", trans_input)

    inp = cv2.resize(img, (self.opt.input_res, self.opt.input_res), interpolation=cv2.INTER_CUBIC)
    # inp = cv2.warpAffine(img, trans_input, 
    #                      (self.opt.input_res, self.opt.input_res),
    #                      flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)

    if self.split == 'train' and not self.opt.no_color_aug:                 # 随机进行图片增强
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_res = self.opt.output_res
    num_joints = self.num_joints
    # cv2.getAffineTransform(np.float32([[5,5], [20, 20], [10, 10]]), np.float32([[10, 10], [40, 40], [20, 20]]))

    # trans_output = cv2.getAffineTransform(np.float32([[0, 0], [width / 2, height / 2], [width / 4, height / 4]]), np.float32([[0, 0], [output_res / 2, output_res / 2], [output_res / 4, output_res / 4]]))
    # trans_output_rot = trans_output 
    # trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
    # trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
    dense_kps = np.zeros((num_joints, 2, output_res, output_res), 
                          dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_res, output_res), 
                               dtype=np.float32)
    ltrb = np.zeros((self.max_objs, 4), dtype=np.float32)
    ltrb_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    wight_mask = np.ones((self.max_objs), dtype=np.float32)
    kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
    hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
      
    # if flipped:
    #   bbox[[0, 2]] = width - bbox[[2, 0]] - 1
    #   pts[:, 0] = width - pts[:, 0] - 1
    #   for e in self.flip_idx:
    #     pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

    
    print("raw bbox", bbox[:2])
    print("raw bbox", bbox[2:])
    # bbox[:2] = affine_transform(bbox[:2], trans_output)
    # bbox[2:] = affine_transform(bbox[2:], trans_output)
    bbox[2] = bbox[2] / width * output_res
    bbox[3] = bbox[3] / height * output_res
    print("after bbox", bbox[:2])
    print("after bbox", bbox[2:])

    bbox = np.clip(bbox, 0, output_res - 1)
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if (h > 0 and w > 0) or (rot != 0):
      radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius)) 
      ct = np.array(
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)       # 人脸的中心坐标

      ct_int = ct.astype(np.int32)                        # 整数化
      # wh[k] = 1. * w, 1. * h                                                    # 2. centernet的方式
      k = 0
      wh[k] = np.log(1. * w), np.log(1. * h)                              # 2. 人脸bbox的高度和宽度,centerface论文的方式

      if self.opt.ltrb: # and pts[2][2] > 0:
        l, t, r, b = ct[0] - bbox[0], ct[1] - bbox[1], bbox[2] - ct[0], bbox[3] - ct[1]
        # l, t, r, b = ct_int[0] - bbox[0], ct_int[1] - bbox[1], bbox[2] - ct_int[0], bbox[3] - ct_int[1]
        ltrb[k] = np.log(1. * l / 2), np.log(1. * t / 2), np.log(1. * r / 2), np.log(1. * b / 2)   

        ltrb_mask[k] = 1

      ind[k] = ct_int[1] * output_res + ct_int[0]         # 人脸bbox在1/4特征图中的索引
      reg[k] = ct - ct_int                                # 3. 人脸bbox中心点整数化的偏差
      reg_mask[k] = 1                                     # 是否需要用于计算误差

      num_kpts = pts[:, 2].sum()                           # 没有关键点标注的时哦
      if num_kpts == 0:                                    # 没有关键点标注的都是比较困难的样本
        # print('没有关键点标注')
        hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
        # reg_mask[k] = 0

      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = self.opt.hm_gauss \
                  if self.opt.mse_loss else max(0, int(hp_radius)) 
      for j in range(num_joints):
        if pts[j, 2] > 0:
          # pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
          print("width, height", width, height)
          print("raw pts", pts[j])
          pts[j, 0] = pts[j, 0] / width * output_res
          pts[j, 1] = pts[j, 0] / height * output_res
          pts = np.clip(pts, 0, output_res - 1)
          print("after pts", pts[j])

          if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
            pts[j, 1] >= 0 and pts[j, 1] < output_res:
            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int                # 4. 关键点相对于人脸bbox的中心的偏差
            kps_mask[k, j * 2: j * 2 + 2] = 1
            pt_int = pts[j, :2].astype(np.int32)                          # 关键点整数化
            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int           # 关键点整数化的偏差
            hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]   # 索引
            hp_mask[k * num_joints + j] = 1                                   # 计算损失的mask
            if self.opt.dense_hp:
              # must be before draw center hm gaussian
              draw_dense_reg(dense_kps[j], hm[cls_id], ct_int, 
                            pts[j, :2] - ct_int, radius, is_offset=True)
              draw_gaussian(dense_kps_mask[j], ct_int, radius)
              print("pt_int", pt_int)
              draw_gaussian(hm_hp[j], pt_int, hp_radius)                    # 1. 关键点高斯map
              # if ann['bbox'][2]*ann['bbox'][3] >= 16.0:                   # 太小的人脸忽略
              #   kps_mask[k, j * 2: j * 2 + 2] = 0
              # print("==== file_name, index ====", file_name, index)
      draw_gaussian(hm[cls_id], ct_int, radius)
      gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                    ct[0] + w / 2, ct[1] + h / 2, 1] + 
                    pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])

    if rot != 0:
      hm = hm * 0 + 0.9999
      reg_mask *= 0
      kps_mask *= 0

    # #  yy add
    hm_vis = 255 - hm_hp[0] * 255
    hm_vis = np.clip(hm_vis, 0, 255)
    hm_vis = np.array(hm_vis, dtype=np.uint8)
    hm_vis = np.expand_dims(hm_vis, axis=-1)
    hm_vis = np.repeat(hm_vis, 3, axis=-1)
    hm_vis = cv2.resize(hm_vis, (width, height)) 
    masked_image =  hm_vis * 0.9 + img * 0.1
    for m in gt_det:
      each_gt_det = m[:4]
      each_gt_det = np.array(each_gt_det, dtype=np.int32)
    cv2.imwrite('debug_whole_body_hm_hp_100.jpg', masked_image)

    #yy add
    hm_vis = 255 - hm[0] * 255
    hm_vis = np.clip(hm_vis, 0, 255)
    hm_vis = np.array(hm_vis, dtype=np.uint8)
    hm_vis = np.expand_dims(hm_vis, axis=-1)
    hm_vis = np.repeat(hm_vis, 3, axis=-1)
    hm_vis = cv2.resize(hm_vis, (width, height)) 
    masked_image =  hm_vis * 0.9 + img * 0.1
    for m in gt_det:
      each_gt_det = m[:4]
      each_gt_det = np.array(each_gt_det, dtype=np.int32)
    cv2.imwrite('debug_whole_body_hm_100.jpg', masked_image)

    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'landmarks': kps, 'hps_mask': kps_mask, 'wight_mask': wight_mask, 'ltrb': ltrb, 'ltrb_mask': ltrb_mask}
    if self.opt.dense_hp:
      dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints, 1, output_res, output_res)
      dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints * 2, output_res, output_res)
      ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
      del ret['hps'], ret['hps_mask']
    if self.opt.reg_offset:
      ret.update({'hm_offset': reg})                  # 人脸bbox中心点整数化的偏差
    if self.opt.hm_hp:
      ret.update({'hm_hp': hm_hp})
    if self.opt.reg_hp_offset:
      ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 48), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta

    return ret


_use_shared_memory = False

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))


def multipose_collate(batch):
  objects_dims = [d.shape[0] for d in batch]
  index = objects_dims.index(max(objects_dims))
  # one_dim = True if len(batch[0].shape) == 1 else False
  res = []
  for i in range(len(batch)):
      tres = np.zeros_like(batch[index], dtype=batch[index].dtype)
      tres[:batch[i].shape[0]] = batch[i]
      res.append(tres)

  return res


def Multiposebatch(batch):
  sample_batch = {}
  for key in batch[0]:
    if key in ['hm', 'input']:
      sample_batch[key] = default_collate([d[key] for d in batch])
    else:
      align_batch = multipose_collate([d[key] for d in batch])
      sample_batch[key] = default_collate(align_batch)

  return sample_batch
