#!/bin/bash
cd ../src
# --load_model ../pretrained/ctdet_coco_dla_1x.pth
CUDA_VISIBLE_DEVICES=5 python main.py --arch dla_34  --task ctdet --dataset facehp --exp_id face_hp_dla --batch_size 8 --lr 5e-4   --gpus 4 --num_workers 4 --num_epochs 140 --lr_step 90,120
