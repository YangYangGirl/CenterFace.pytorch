from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from external.nms import soft_nms_39
from models.decode import multi_pose_decode, centerface_decode
from models.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr, flip_lr_off_body
from utils.image import get_affine_transform
from utils.post_process import whole_body_post_process

from .base_detector import BaseDetector

class WholeBodyDetector(BaseDetector):
  def __init__(self, opt):
    super(WholeBodyDetector, self).__init__(opt)
    self.flip_idx = opt.flip_idx
    

  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = _sigmoid(output['hm'])
      if self.opt.hm_hp and not self.opt.mse_loss:
        output['hm_hp'] = _sigmoid(output['hm_hp'])

      reg = output['hm_offset'] if self.opt.reg_offset else None
      hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      if self.opt.flip_test:
        output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        if self.opt.ltrb:
          #output['ltrb'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
          print("to do!")
        else:
          output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off_body(output['hps'][1:2], self.flip_idx)) / 2
        hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                if hm_hp is not None else None
        reg = reg[0:1] if reg is not None else None
        hp_offset = hp_offset[0:1] if hp_offset is not None else None
      
      dets = multi_pose_decode(
        output['hm'], wh=output['wh'], kps=output['hps'],
        reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)


    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = whole_body_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 47)             # 关键点数+5=26
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]

    for k in results.keys():
      results[k] = results[k].tolist()
    return results

    # results = {}
    # results[1] = np.concatenate(
    #     [detection[1] for detection in detections], axis=0).astype(np.float32)
    # if self.opt.nms or len(self.opt.test_scales) > 1:
    #   print("self.opt.test_scales", self.opt.test_scales)
    #   print("nms", self.opt.nms)
    #   soft_nms_39(results[1], Nt=0.5, method=2)
    # for k in results.keys():
    #   results[k] = results[k].tolist()
    # return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:47] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')
  
  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='multi_pose')
    for bbox in results[1]:
      if bbox[4] > 0.2:#self.opt.vis_thresh:
        # debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
        debugger.add_whole_body_points(bbox[5:47], img_id='multi_pose')

        debug_image = image
        points = bbox[5:47]
        points = np.array(points, dtype=np.int32).reshape(21, 2)
        for j in range(21):
          debug_image = cv2.circle(debug_image,
                    (points[j, 0], points[j, 1]), 3, (0, 0, 255), 1)
        # for person pose edege show
        # for j, e in enumerate(self.edges):
        #   if points[e].min() > 0:
        #     self.imgs[img_id] = cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
        #                   (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
        #                   lineType=cv2.LINE_AA)
        cv2.imwrite("yxyy-body-1-0.51.jpg", debug_image)

    debugger.show_all_imgs(pause=self.pause)

  def return_results(self, debugger, image, results):
    debugger.add_img(image, img_id='multi_pose')
    for bbox in results[1]:
      if bbox[4] > 0.2:#self.opt.vis_thresh:
        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
        debugger.add_whole_body_points(bbox[5:47], img_id='multi_pose')
    return debugger.return_img(img_id='multi_pose')