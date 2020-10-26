#!/bin/bash
cd ../src
CUDA_VISIBLE_DEVICES=0 python main.py --task multi_pose --arch mobilev2_10 --dataset facehp --exp_id face_hp_mobilev2_10 --batch_size 8 --master_batch 8 --lr 5e-4   --gpus 0 --num_workers 4 --num_epochs 140 --lr_step 90,120
