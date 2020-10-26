#!/bin/bash
cd ../src
# --load_model ../pretrained/ctdet_coco_dla_1x.pth
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --load_model '' --arch mobilev2_5 --task ctdet --dataset face --exp_id 1002_opt2_face_hp_mobilev2_5 --batch_size 8 --lr 1.6e-4   --gpus 3 --num_workers 4 --num_epochs 200 --lr_step 90,120
