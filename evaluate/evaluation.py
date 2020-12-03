#-*-coding:utf-8-*-
from __future__ import division
"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""
import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed
import cv2


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]      
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()  #取出与预测框重叠度最高的gt框
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1     #有预测框被阈值错误过滤，未检测出来
                proposal_list[h] = -1         #该预测框被阈值过滤
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):
        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def vis_badcase(pred_info, gt_boxes, ori_image_path, target_path="vis-0.2"):
    have_hitted=[]
    gt_have_hitted=[]
    ori_image = cv2.imread(ori_image_path)
    pred_image = cv2.imread(ori_image_path)
    h, w, c = ori_image.shape
    event_name = ori_image_path.split('/', 7)[-2]
    pic_name = ori_image_path.split('/', 7)[-1]
    hp_img = cv2.resize(cv2.imread('./heatmap-0.3/' + ori_image_path.split('/', 6)[-1]), (w, h))
    hp_pure_img = cv2.resize(cv2.imread('./heatmap-0.3/' + event_name + '/' + 'pure_' + pic_name), (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    iou_threshold = 0.4
    score_threshold = 0.2
    keep_idx = pred_info[:, 4] > score_threshold
    pred_info = pred_info[keep_idx]
    

    res = {'GtNums': 0, 'TP': 0, 'FP': 0, 'TN': 0, 'PNums': 0, "recall": 0, "Precision": 0}

    for gt_idx, obj in enumerate(gt_boxes):
        x1_, y1_, w_, h_ = obj[0], obj[1], obj[2], obj[3] 
        x2_ = x1_ + w_
        y2_ = y1_ + h_
        res["GtNums"] += 1
        ori_image = cv2.rectangle(ori_image, (int(x1_), int(y1_)), (int(x2_), int(y2_)), (255, 255, 0), 2)

        match_index = -1
        max_iou = 0
        match_box =[]
        
        for idx in range(pred_info.shape[0]):
            x1, y1, w, h, s = pred_info[idx]
            x2 = x1 + w
            y2 = y1 + h
            pred_image = cv2.rectangle(pred_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            if idx not in have_hitted:
                iou = compute_iou([x1_, y1_, x2_, y2_], [x1, y1, x2, y2]) 
                if iou > iou_threshold:
                    if iou > max_iou:
                        max_iou = iou
                        match_index = idx

            if match_index != -1:
                res["TP"] += 1
                have_hitted.append(match_index)
                gt_have_hitted.append(gt_idx)
                x1, y1, w, h, s = pred_info[match_index]
                x2 = x1 + w
                y2 = y1 + h
                pred_image = cv2.putText(pred_image, "tp:" + " "+ str(s)[:3], (int(x1), int(y1)), font, 1, (255, 0, 0), 2)
                pred_image = cv2.rectangle(pred_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    for j in range(pred_info.shape[0]):
        x1, y1, w, h, s = pred_info[j]
        x2 = x1 + w
        y2 = y1 + h
        if j not in have_hitted:
            res["FP"] += 1
            pred_image = cv2.putText(pred_image,
                                "fp:" + " "+ str(s)[:3],
                                (int(x2), int(y2)), font, 1, (0, 0, 255), 2)
            pred_image = cv2.rectangle(pred_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    for j in range(gt_boxes.shape[0]):
        x1, y1, w, h = gt_boxes[j]
        x2 = x1 + w
        y2 = y1 + h
        if j not in gt_have_hitted:
            res["TN"] += 1
            # ori_image = cv2.putText(ori_image,
            #                     "tn:" +" ",
            #                     (int(x2), int(y2)), font, 1, (0, 255, 0), 2)
            ori_image = cv2.rectangle(ori_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if not os.path.exists(target_path):
            os.mkdir(target_path)

        cv2.imwrite(target_path + "/" + os.path.basename(ori_image_path), np.concatenate((np.hstack((pred_image, ori_image)) , np.hstack((hp_img, hp_pure_img))), axis=0))

    if res["GtNums"] == 0 or res["PNums"] == 0:
        # print("=========== r['GtNums', 'PNums'] =========", res["GtNums"], res["PNums"])
        res["recall"] = -1
        res["Precision"] = -1

    else:
        res["recall"] = res["TP"] / res["GtNums"] 
        res["Precision"] = res["TP"] / res["PNums"] 
        print(res) 

        return None


def evaluation(pred, gt_path, all, iou_thresh=0.4):
    pred = get_preds(pred)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    
    if not all:
        aps = []
        for setting_id in range(3):
            # different setting
            gt_list = setting_gts[setting_id]
            count_face = 0
            pr_curve = np.zeros((thresh_num, 2)).astype('float')
            # [hard, medium, easy]
            pbar = tqdm.tqdm(range(event_num))   # event_num = 61
            error_count = 0
            for i in pbar:
                pbar.set_description('Processing {}'.format(settings[setting_id]))
                event_name = str(event_list[i][0][0])
                img_list = file_list[i][0]
                pred_list = pred[event_name]  
                sub_gt_list = gt_list[i][0]
                gt_bbx_list = facebox_list[i][0]
                
                for j in range(len(img_list)):
                    try:
                        pred_info = pred_list[str(img_list[j][0][0])] 
                        ori_image_path = "../data/widerface/retinaface_gt_v1.1/val/images/" + str(event_list[i][0][0]) + '/' + str(img_list[j][0][0]) + '.jpg'
                    except:
                        error_count+=1
                        continue
                    gt_boxes = gt_bbx_list[j][0].astype('float')

                    # vis_badcase(pred_info, gt_boxes, ori_image_path)

                    keep_index = sub_gt_list[j][0]
                    count_face += len(keep_index)
                    if len(gt_boxes) == 0 or len(pred_info) == 0:
                        continue
                    ignore = np.zeros(gt_boxes.shape[0])
                    if len(keep_index) != 0:
                        ignore[keep_index-1] = 1
                    pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                    _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                    pr_curve += _img_pr_info
            pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

            propose = pr_curve[:, 0]
            recall = pr_curve[:, 1]

            ap = voc_ap(recall, propose)
            aps.append(ap)

        print("==================== Results ====================")
        print("Easy   Val AP: {}".format(aps[0]))
        print("Medium Val AP: {}".format(aps[1]))
        print("Hard   Val AP: {}".format(aps[2]))
        print("=================================================")
    else:
        aps = []
        # different setting
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')  #  control calcultate how many samples
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        error_count = 0
        for i in pbar:
            pbar.set_description('Processing {}'.format("all"))
            # print("event_list is: ",event_list)
            event_name = str(event_list[i][0][0])  #  '0--Parade', '1--Handshaking'
            img_list = file_list[i][0]
            pred_list = pred[event_name]  #  每个文件夹的所有检测结果
            sub_gt_list = [ setting_gts[0][i][0], setting_gts[1][i][0], setting_gts[2][i][0] ]

            gt_bbx_list = facebox_list[i][0]
            for j in range(len(img_list)):
                try:
                    pred_info = pred_list[str(img_list[j][0][0])]  #   # str(img_list[j][0][0] 是每个folder下面的图片名字
                except:
                    error_count+=1
                    continue

                gt_boxes = gt_bbx_list[j][0].astype('float')
                temp_i = []
                for ii in range(3):
                    if len(sub_gt_list[ii][j][0])!=0:
                        temp_i.append(ii)
                if len(temp_i)!=0:
                    keep_index = np.concatenate(tuple([sub_gt_list[xx][j][0] for xx in temp_i]))
                else:
                    keep_index = []
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0]) # #  no ignore
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info

        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

        print("==================== Results ====================")
        print("All Val AP: {}".format(aps[0]))
        print("=================================================")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default='/home/deepblue/deepbluetwo/chenjun/4_face_detect/centerface/output/widerface')
    parser.add_argument('-g', '--gt', default='./ground_truth')
    parser.add_argument('--all', help='if test all together', action='store_true')

    args = parser.parse_args()
    evaluation(args.pred, args.gt, args.all)












