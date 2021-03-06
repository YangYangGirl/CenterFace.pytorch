# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F

class OhemLoss(nn.Module):
  def __init__(self):
      super(OhemLoss, self).__init__()

  def forward(self, pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = - torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = - torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()

    pos_loss = torch.flatten(pos_loss)
    neg_loss = torch.flatten(neg_loss)

    pos_loss, pos_idx = torch.sort(pos_loss, descending=True)
    neg_loss, neg_idx = torch.sort(neg_loss, descending=True)

    neg_pos_ratio = 3
    keep_num = int(num_pos.item())
    
    keep_pos_idx = pos_idx[:keep_num]
    keep_neg_idx = neg_idx[:keep_num * neg_pos_ratio]
  
    keep_pos_loss = pos_loss[keep_pos_idx]
    keep_neg_loss = neg_loss[keep_neg_idx]

    loss = (keep_pos_loss.sum() + keep_neg_loss.sum())/ (num_pos * (neg_pos_ratio + 1))

    return loss


class GIoULoss(nn.Module):
  def __init__(self):
      super(GIoULoss, self).__init__()
      self.shift = None

  def forward(self, output, gt_mask, ind, target, wight_=None, pre_off=None, target_off=None):
      bs, objs, h, w = output.shape
      gt = torch.zeros_like(output) # [8, 4, 200, 200]
      mask = torch.zeros(bs, 4, h, w)
      pred = output
      weight = torch.zeros(bs, 4, h, w)

      #ind[k] = ct_int[1] * output_res + ct_int[0]         # 人脸bbox在1/4特征图中的索引  ct_int w,h
      ind_x = ind % 200  # [8, 32]
      ind_y = ind // 200
      
      for b_inx in range(bs):
        for o_index in range(objs):
          gt[b_inx, :, ind_x[b_inx][o_index], ind_y[b_inx][o_index]] = target[b_inx][o_index]
          mask[b_inx, :, ind_x[b_inx][o_index], ind_y[b_inx][o_index]] = gt_mask[b_inx][o_index]
          weight[b_inx, :, ind_x[b_inx][o_index], ind_y[b_inx][o_index]] = wight_[b_inx][o_index]

      if self.shift is None:
          x = torch.arange(0, w, device=pred.device)
          y = torch.arange(0, h, device=pred.device)
          shift_y, shift_x = torch.meshgrid(y, x)
          self.shift = torch.stack((shift_x, shift_y), dim=0).float()   # 2, h, w

      pred_boxes = torch.cat((
          self.shift - pred[:, [0, 1]],
          self.shift + pred[:, [2, 3]]
      ), dim=1).permute(0, 2, 3, 1)  # b, 4, h, w   to   b, h, w, 4

      # mask = mask.permute(0, 2, 3, 1)
      gt_boxes = torch.cat((
          self.shift - gt[:, [0, 1]],
          self.shift + gt[:, [2, 3]]
      ), dim=1).permute(0, 2, 3, 1)  # b, 4, h, w   to   b, h, w, 4

      mask = mask.permute(0, 2, 3, 1)
      weight = weight.permute(0, 2, 3, 1)
      pred_boxes = pred_boxes[mask==1].view(-1, 4)
      gt_boxes = gt_boxes[mask==1].view(-1, 4)
      avg_factor = torch.sum(weight) / 4
      weight = weight[mask==1][0]

      # max x, max y
      lt = torch.max(pred_boxes[:, :2], gt_boxes[:, :2])

      # min r, min b
      rb = torch.min(pred_boxes[:, 2:], gt_boxes[:, 2:])

      wh = (rb - lt + 1).clamp(0) # n, 2

      enclose_lt = torch.min(pred_boxes[:, :2], gt_boxes[:, :2])
      enclose_rb = torch.max(pred_boxes[:, 2:], gt_boxes[:, 2:])
      enclose_wh = (enclose_rb - enclose_lt + 1).clamp(0)  # n, 2
      enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
      overlap = wh[:, 0] * wh[:, 1]

      pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1)
      gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
      ious = overlap / (pred_area + gt_area - overlap)

      u = pred_area + gt_area - overlap
      gious = ious - (enclose_area - u) / enclose_area
      iou_distance = 1 - gious
      return torch.sum(iou_distance * weight) / avg_factor


def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask, wight_=None):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask

  if wight_ is not None:
      wight_ = wight_.unsqueeze(2).expand_as(gt_regr).float()
      regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduce=False)
      regr_loss *= wight_
      regr_loss = regr_loss.sum()
  else:
      regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)

  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)


class SlowNegLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(SlowNegLoss, self).__init__()
    self._slow_neg_loss = _slow_neg_loss

  def forward(self, out, target):
    return self._slow_neg_loss(out, target)


class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target, wight_=None):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask, wight_)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
