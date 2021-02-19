#!/bin/bash
cd ../src

CUDA_VISIBLE_DEVICES=0,1 python main.py --slow_neg_loss --hm_weight 1 --not_hm_hp --off_weight 1 --wh_weight 0.1 --lm_weight 0 --hm_hp_weight 0 --val_intervals 1 --head_conv 24 --task ctdet --arch mobilev3_10 --dataset wholebody --input_w 224 --input_h 224 --exp_id 0108_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_224_224_bs16_5e-4  --batch_size 32 --master_batch 16 --lr 5e-4 --gpus 0,1 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=3 python main.py --slow_neg_loss --hm_weight 1 --not_hm_hp --off_weight 1 --wh_weight 0.1 --lm_weight 0 --hm_hp_weight 0 --val_intervals 5 --head_conv 24 --task ctdet --arch mobilev3_10 --dataset wholebody --input_w 800 --input_h 800 --exp_id 0111_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_bifpn_only_box_obj600_800_800_bs8_5e-4  --batch_size 8 --master_batch 8 --lr 5e-4 --gpus 0 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=0,1 python main.py --slow_neg_loss --hm_weight 1 --not_hm_hp --off_weight 1 --wh_weight 0.1 --lm_weight 0 --hm_hp_weight 0 --val_intervals 5 --head_conv 24 --task ctdet --arch mobilev3_10 --dataset wholebody --input_w 224 --input_h 224 --exp_id 0108_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_224_224_bs16_5e-4  --batch_size 32 --master_batch 16 --lr 5e-4 --gpus 0,1 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=2,3 python main.py --hm_weight 1 --not_hm_hp --off_weight 1 --wh_weight 0.1 --lm_weight 0 --hm_hp_weight 0 --val_intervals 5 --head_conv 24 --task ctdet --arch mobilev3_10 --dataset wholebody --input_w 800 --input_h 800 --exp_id 0110_0.75_normal_radius_adam_ctdet_faster_mobilev3_10_fpn_only_box_obj600_800_800_bs8_5e-4  --batch_size 24 --master_batch 12 --lr 5e-4 --gpus 0,1 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=0 python main.py --slow_neg_loss --hm_weight 1 --not_hm_hp --off_weight 1 --wh_weight 0.1 --lm_weight 0 --hm_hp_weight 0 --val_intervals 5 --head_conv 24 --task ctdet --arch mobilev3_10 --dataset wholebody --input_w 576 --input_h 576 --exp_id 0115_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_576_576_bs16_5e-4  --batch_size 16 --master_batch 8 --lr 5e-4 --gpus 0 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=1 python main.py --hm_weight 1 --not_hm_hp --off_weight 1 --wh_weight 0.1 --lm_weight 0 --hm_hp_weight 0 --val_intervals 5 --head_conv 24 --task ctdet --arch mobilev3_10 --dataset wholebody --input_w 800 --input_h 800 --exp_id 0105_seperate_normal_radius_adam_ctdet_mobilev3_10_fpn_obj600_800_800_bs8_5e-4 --batch_size 8 --master_batch 8 --lr 5e-4 --gpus 0 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=2 python main.py --slow_neg_loss --hm_weight 1 --not_hm_hp --off_weight 1 --wh_weight 0.1 --lm_weight 0 --hm_hp_weight 0 --val_intervals 5 --head_conv 24 --task ctdet --arch mobilev3_10 --dataset wholebody --input_w 800 --input_h 800 --exp_id 1231_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4 --load_model '../exp/ctdet/1231_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_best_60.pth' --batch_size 12 --master_batch 12 --lr 5e-5 --gpus 0 --num_workers 4  --lr_step 90,120

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0111_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_bifpn_only_box_obj600_800_800_bs8_5e-4/model_last.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0110_0.75_normal_radius_adam_ctdet_faster_mobilev3_10_fpn_only_box_obj600_800_800_bs8_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 224 --input_h 224 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/1231_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 224 --input_h 224 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0110_0.75_normal_radius_adam_ctdet_faster_mobilev3_10_fpn_only_box_obj600_800_800_bs8_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 224 --input_h 224 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0108_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_224_224_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 224 --input_h 224 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0107_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_224_224_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 256 --input_h 256 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0107_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_256_256_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 576 --input_h 576 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0107_0.75_normal_radius_adam_ctdet_faster_mobilev3_10_fpn_only_box_obj600_800_800_bs8_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 384 --input_h 384 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0107_0.75_slow_neg_loss_normal_radius_adam_ctdet_faster_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0105_seperate_normal_radius_adam_ctdet_mobilev3_10_fpn_obj600_800_800_bs8_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0105_mish_normal_radius_adam_ctdet_mobilev3_10_fpn_obj600_800_800_bs8_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/1231_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_best_60_5.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0111_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_bifpn_only_box_obj600_800_800_bs8_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/1231_normal_radius_adam_ctdet_mobilev3_10_pan_only_box_obj600_800_800_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0105_seperate_normal_radius_adam_ctdet_mobilev3_10_fpn_obj600_800_800_bs8_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/1231_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_last.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 800 --input_h 800 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/1231_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 576 --input_h 576 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0107_1_normal_radius_adam_ctdet_faster_mobilev3_10_fpn_only_box_obj600_576_576_bs8_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 384 --input_h 384 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0107_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_384_384_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 256 --input_h 256 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0107_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_256_256_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

# CUDA_VISIBLE_DEVICES=0 python test.py --not_hm_hp --not_reg_hp_offset --task ctdet --input_w 224 --input_h 224 --arch mobilev3_10 --head_conv 24 --load_model ../exp/ctdet/0108_slow_neg_loss_normal_radius_adam_ctdet_mobilev3_10_fpn_only_box_obj600_224_224_bs16_5e-4/model_best.pth --dataset wholebody --keep_res --resume --flip_test

