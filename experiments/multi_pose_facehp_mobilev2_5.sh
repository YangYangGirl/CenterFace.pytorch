#!/bin/bash
cd ../src
# CUDA_VISIBLE_DEVICES=4 python main.py --arch mobilev2_5 --dataset facehp --exp_id face_hp_mobilev2_5 --batch_size 8 --master_batch 8 --lr 5e-4  --gpus 5 --num_workers 4 --num_epochs 140 --lr_step 90,120
CUDA_VISIBLE_DEVICES=7 python main.py --task multi_pose --arch mobilev2_5 --dataset facehp --exp_id face_hp_mobilev2_5_nohp --batch_size 8 --master_batch 8 --lr 5e-4   --gpus 6 --num_workers 4 --num_epochs 140 --lr_step 90,120
