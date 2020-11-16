#!/bin/bash
cd ../src
# --load_model '../exp/multi_pose/1031_face_hp_mobilev2_10_640_640/model_140.pth'
# CUDA_VISIBLE_DEVICES=3 python main.py  --load_model '../exp/multi_pose/1031_face_hp_mobilev2_10_640_640/model_140.pth' --task multi_pose --arch mobilev2_10 --dataset facehp --exp_id 1031_face_hp_mobilev2_10_800_800 --batch_size 8 --master_batch 8 --lr 3.125e-5   --gpus 3 --num_workers 4 --num_epochs 200 --lr_step 90,120
CUDA_VISIBLE_DEVICES=3 python main.py --task multi_pose --arch shufflenetv2_10 --dataset facehp --exp_id 1113_face_hp_shufflenetv2_10_800_800_bs12_head64_5e-4 --batch_size 12 --master_batch 12 --lr 5e-4  --gpus 3 --num_workers 4 --num_epochs 300 --lr_step 90,120