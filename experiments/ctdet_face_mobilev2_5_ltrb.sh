#!/bin/bash
cd ../src
# --load_model ../pretrained/ctdet_coco_dla_1x.pth
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --ltrb --load_model '' --arch mobilev2_5 --task ctdet --dataset facehp --exp_id 1015_opt2_facehp_mobilev2_5_nose --batch_size 8 --lr 1.6e-4   --gpus 3 --num_workers 4 --num_epochs 200 --lr_step 90,120
