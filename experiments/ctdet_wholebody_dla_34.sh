#!/bin/bash
cd ../src

# CUDA_VISIBLE_DEVICES=2,3 python main.py --hm_weight 1 --not_hm_hp --off_weight 1 --wh_weight 0.1 --lm_weight 0 --hm_hp_weight 0 --val_intervals 5 --head_conv 24 --task ctdet --arch dla_34 --dataset wholebody --input_w 800 --input_h 800 --exp_id 0104_normal_radius_adam_ctdet_dla_34_fpn_only_box_obj600_800_800_bs16_5e-4 --batch_size 16 --master_batch 8 --lr 5e-4 --gpus 0,1 --num_workers 4  --lr_step 90,120

CUDA_VISIBLE_DEVICES=1 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch dla_34  --head_conv 24 --load_model ../exp/ctdet/0104_normal_radius_adam_ctdet_dla_34_fpn_only_box_obj600_800_800_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test
