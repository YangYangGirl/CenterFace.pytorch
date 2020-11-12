#!/bin/bash
cd ../src
CUDA_VISIBLE_DEVICES=2 python main.py --arch mobilev2_5 --dataset facehp --exp_id 1030_s_face_hp_mobilev2_5_640_640 --batch_size 8 --master_batch 8 --lr 5e-4  --gpus 2 --num_workers 4 --num_epochs 140 --lr_step 90,120
# CUDA_VISIBLE_DEVICES=3 python main.py --load_model ../exp/multi_pose/1028_facehp_mobilev2_5_640_640/model_80.pth --task multi_pose --arch mobilev2_5 --dataset facehp --exp_id 1029_ft640_s_e80_30_60_facehp_mobilev2_5_800_800  --batch_size 8 --master_batch 8 --lr 5e-4   --gpus 3 --num_workers 4 --num_epochs 200 --lr_step 30,60
