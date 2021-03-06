#!/bin/bash
cd ../src
# --load_model '../exp/multi_pose/1031_face_hp_mobilev2_10_640_640/model_140.pth'
# CUDA_VISIBLE_DEVICES=3 python main.py  --load_model '../exp/multi_pose/1031_face_hp_mobilev2_10_640_640/model_140.pth' --task multi_pose --arch mobilev2_10 --dataset facehp --exp_id 1031_face_hp_mobilev2_10_800_800 --batch_size 8 --master_batch 8 --lr 3.125e-5   --gpus 3 --num_workers 4 --num_epochs 200 --lr_step 90,120
# --load_model '../exp/multi_pose/1106_face_hp_mobilev2_10_640_640_bs12_5e-4/model_140.pth' 
CUDA_VISIBLE_DEVICES=0 python main.py --val_intervals 5 --head_conv 24 --task multi_pose --arch mobilev3_10 --dataset facehp --exp_id 1212_obj600_face_hp_mobilev3_10_800_800_bs8_5e-4 --batch_size 8 --master_batch 8 --lr 5e-4  --gpus 0 --num_workers 4 --num_epochs 300 --lr_step 90,120
