#!/bin/bash
cd ../src
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --not_hm_hp --not_reg_hp_offset --val_intervals 5 --head_conv 24 --task multi_pose_whole --arch mobilev3_10 --dataset wholebody --exp_id 1220_only_hm_sample_obj600_whole_body_mobilev3_10_800_800_bs48_5e-5 --batch_size 48 --master_batch 16 --lr 5e-5  --gpus 0,1,2 --num_workers 4 --num_epochs 300 --lr_step 90,120

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset  --task multi_pose_whole --arch mobilev3_10 --head_conv 24 --load_model ../exp/multi_pose_whole/1220_only_hm_sample_obj600_whole_body_mobilev3_10_800_800_bs48_5e-5/model_last.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=1 python test.py  --task multi_pose_whole --exp_id 1221_obj600_whole_body_mobilev3_10_800_800_bs16_5e-5 --arch mobilev3_10 --head_conv 24 --load_model ../exp/multi_pose_whole/1221_obj600_whole_body_mobilev3_10_800_800_bs16_5e-5/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=1 python test.py  --lm_weight 0 --hm_hp_weight 0 --not_hm_hp --not_reg_hp_offset  --task multi_pose_whole --exp_id 1220_only_hm_sample_obj600_whole_body_mobilev3_10_800_800_bs48_5e-5 --arch mobilev3_10 --head_conv 24 --load_model ../exp/multi_pose_whole/1220_only_hm_sample_obj600_whole_body_mobilev3_10_800_800_bs48_5e-5/model_best.pth --dataset wholebody --keep_res --resume --flip_test

CUDA_VISIBLE_DEVICES=1,2 python main.py --hm_weight 0 --off_weight 0 --wh_weight 0  --lm_weight 1 --hm_hp_weight 1  --not_reg_hp_offset --val_intervals 5 --head_conv 24 --task multi_pose_crop --arch mobilev3_10 --dataset cropbody --exp_id 1225_mobilev3_10_only_box_obj600_whole_body_crop_800_800_bs24_5e-5 --batch_size 24 --master_batch 12 --lr 5e-5  --gpus 0 --num_workers 4 --num_epochs 300 --lr_step 90,120
# CUDA_VISIBLE_DEVICES=1,2 python main.py --lm_weight 0 --hm_hp_weight 0 --not_hm_hp --not_reg_hp_offset  --num_epochs 1 --val_intervals 5 --head_conv 24 --task multi_pose_whole --arch mobilev3_10 --dataset wholebody --exp_id 1225_mobilev3_10_only_box_obj600_whole_body_800_800_bs24_5e-6 --batch_size 2 --master_batch 1 --lr 5e-6  --gpus 0 --num_workers 4  --lr_step 90,120
