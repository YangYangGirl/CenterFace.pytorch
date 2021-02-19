#!/bin/bash
cd ../src

CUDA_VISIBLE_DEVICES=2,3 python main.py --load_model '' --not_rand_crop --input_w 128 --input_h 128 --hm_hp_weight 1 --hm_weight 1 --off_weight 1 --wh_weight 0.1 --lm_weight 1 --val_intervals 50000 --head_conv 24 --task multi_pose_crop --arch mobilev3_10  --dataset cropbody --exp_id 0207_nomse_adam_mobilev3_10_obj600_whole_body_crop_128_128_bs16_5e-4 --batch_size 16 --master_batch 8 --lr 5e-4  --gpus 2,3 --num_workers 4 --num_epochs 50000 --lr_step 90,120