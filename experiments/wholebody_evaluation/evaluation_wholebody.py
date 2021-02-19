from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from pycocotools.coco import COCO
import sys
sys.path.append("../experiments/wholebody_evaluation")

from myeval_body import MYeval_body
from myeval_foot import MYeval_foot
from myeval_face import MYeval_face
from myeval_lefthand import MYeval_lefthand
from myeval_righthand import MYeval_righthand
from myeval_wholebody import MYeval_wholebody

def parse_args():
    parser = argparse.ArgumentParser(description='COCO-WholeBody mAP Evaluation')
    parser.add_argument('--res_file',
                        default='',
                        help='tha path to result file',
                        required=False,
                        type=str)
    parser.add_argument('--gt_file',
                        default='../data/coco/annotations/coco_wholebody_val_v1.0.json',
                        help='tha path to gt file',
                        required=False,
                        type=str)
    args = parser.parse_args()
    return args

def test_body(coco,coco_dt):
    print('body mAP ----------------------------------')
    coco_eval = MYeval_body(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_foot(coco,coco_dt):
    print('foot mAP ----------------------------------')
    coco_eval = MYeval_foot(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_face(coco,coco_dt):
    print('face mAP ----------------------------------')
    coco_eval = MYeval_face(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_lefthand(coco,coco_dt):
    print('lefthand keypoints mAP ----------------------------------')
    coco_eval = MYeval_lefthand(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_lefthand_bbox(coco,coco_dt):
    print('lefthand bbox mAP ----------------------------------')
    coco_eval = MYeval_lefthand(coco, coco_dt, 'bbox')
    # coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_righthand(coco,coco_dt):
    print('righthand keypoints mAP ----------------------------------')
    coco_eval = MYeval_righthand(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_righthand_bbox(coco,coco_dt):
    print('righthand bbox mAP ----------------------------------')
    coco_eval = MYeval_righthand(coco, coco_dt, 'bbox')
    # coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_wholebody(coco,coco_dt):
    print('wholebody mAP ----------------------------------')
    coco_eval = MYeval_wholebody(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def eval_lefthand_datasets(res_path):
    coco = COCO('../data/coco/annotations/coco_wholebody_val_v1.0.json')
    print('Testing: {}'.format(res_path), flush=True)
    coco_dt = coco.loadRes(res_path)

    # test_body(coco,coco_dt)
    # test_foot(coco, coco_dt)
    # test_face(coco, coco_dt)
    test_lefthand_bbox(coco, coco_dt)
    test_righthand_bbox(coco, coco_dt)
    test_lefthand(coco, coco_dt)
    test_righthand(coco, coco_dt)
    # test_righthand(coco, coco_dt)
    # test_wholebody(coco, coco_dt)

def main():
    args = parse_args()
    coco = COCO(args.gt_file)
    print('Testing: {}'.format(args.res_file), flush=True)
    coco_dt = coco.loadRes(args.res_file)

    # test_body(coco,coco_dt)
    # test_foot(coco, coco_dt)
    # test_face(coco, coco_dt)
    test_lefthand_bbox(coco, coco_dt)
    test_lefthand(coco, coco_dt)
    # test_righthand(coco, coco_dt)
    # test_wholebody(coco, coco_dt)

if  __name__ == '__main__':
   main()
