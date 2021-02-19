#!/bin/bash
cd ../src

CUDA_VISIBLE_DEVICES=2,3 python main.py --num_epochs 300 --exp_id 0219_coco_hp_with_keypoints_mobilev3_10_pan_800_800 --input_w 800 --input_h 800 --val_intervals 5000 --head_conv 24 --task multi_pose --arch mobilev3_10 --dataset coco_hp --batch_size 16 --master_batch 8 --lr 5e-5  --gpus 0,1 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=1 python test.py --task multi_pose_whole --arch mobilev3_10 --load_model ../exp/multi_pose_whole/dla/model_best.pth --head_conv 24  --dataset wholebody --keep_res --resume --flip_test
