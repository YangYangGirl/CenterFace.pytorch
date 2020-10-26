#!/bin/bash
cd ../src
# --load_model ../pretrained/ctdet_coco_dla_1x.pth
CUDA_VISIBLE_DEVICES=2 python main.py --load_model '' --arch mobilev2_10 --task ctdet --dataset face --exp_id opt2_face_hp_mobilev2_10 --batch_size 8 --lr 5e-4   --gpus 0 --num_workers 4 --num_epochs 200 --lr_step 90,120
