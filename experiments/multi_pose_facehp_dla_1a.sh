#!/bin/bash
cd ../src
CUDA_VISIBLE_DEVICES=1,3 python main.py  --task multi_pose --load_model ../exp/ctdet/dla/model_best.pth --arch dla_34 --dataset facehp --exp_id face_hp_dla_1007 --batch_size 16 --master_batch 8 --lr 10e-4   --gpus 1,3 --num_workers 4 --num_epochs 140 --lr_step 45,60
