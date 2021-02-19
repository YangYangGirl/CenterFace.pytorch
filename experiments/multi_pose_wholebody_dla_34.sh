#!/bin/bash
cd ../src

CUDA_VISIBLE_DEVICES=2,3 python main.py --mse_loss --num_epochs 300 --val_intervals 5 --head_conv 24 --task multi_pose_whole --arch dla_34 --dataset wholebody --load_model '' --exp_id 0104_mse_small_radius_ignore_small_dla_obj600_whole_body_800_800_bs16_5e-5 --batch_size 16 --master_batch 8 --lr 5e-5  --gpus 0,1 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=1,2 python main.py --num_epochs 300 --val_intervals 5 --head_conv 24 --task multi_pose_whole --arch mobilev3_10 --dataset wholebody --load_model ../exp/ctdet/1231_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_best_60_5.pth --exp_id 0102_mobilev3_10_obj600_whole_body_800_800_bs16_5e-5 --batch_size 16 --master_batch 8 --lr 5e-5  --gpus 0,1 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=3 python main.py --mse_loss --num_epochs 300 --val_intervals 5 --head_conv 24 --task multi_pose_whole --arch mobilev3_10 --dataset wholebody --load_model ../exp/ctdet/1231_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_best_60_5.pth --exp_id 0102_mse_mobilev3_10_obj600_whole_body_800_800_bs8_5e-5 --batch_size 8 --master_batch 8 --lr 5e-5  --gpus 0 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=0 python test.py --task multi_pose_whole --arch mobilev3_10 --load_model ../exp/multi_pose_whole/0102_mobilev3_10_obj600_whole_body_800_800_bs16_5e-5/model_best.pth --head_conv 24  --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --task multi_pose_whole --arch mobilev3_10 --load_model ../exp/multi_pose_whole/0102_mse_mobilev3_10_obj600_whole_body_800_800_bs8_5e-5/model_best.pth --head_conv 24  --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --task multi_pose_whole --arch mobilev3_10 --load_model ../exp/multi_pose_whole/0103_mse_ignore_small_mobilev3_10_obj600_whole_body_800_800_bs8_5e-5/model_best.pth --head_conv 24  --dataset wholebody --keep_res --resume --flip_test