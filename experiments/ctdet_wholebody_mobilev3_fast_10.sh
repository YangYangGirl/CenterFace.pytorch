#!/bin/bash
cd ../src

CUDA_VISIBLE_DEVICES=3 python main.py --slow_neg_loss --hm_weight 1 --not_hm_hp --off_weight 1 --wh_weight 0.1 --lm_weight 0 --hm_hp_weight 0 --val_intervals 5 --head_conv 24 --task ctdet --arch mobilev3fast_10 --dataset wholebody --input_w 800 --input_h 800 --exp_id 0108_0.75_slow_neg_loss_normal_radius_adam_ctdet_faster_mobilev3fast_10_fpn_only_box_obj600_800_800_bs12_5e-4  --batch_size 12 --master_batch 12 --lr 5e-5 --gpus 0 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3fast_10 --head_conv 24 --load_model ../exp/ctdet/0108_0.75_slow_neg_loss_normal_radius_adam_ctdet_faster_mobilev3fast_10_fpn_only_box_obj600_800_800_bs12_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3fast_10 --head_conv 24 --load_model ../exp/ctdet/0107_1_slow_neg_loss_normal_radius_adam_ctdet_faster_mobilev3fast_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

