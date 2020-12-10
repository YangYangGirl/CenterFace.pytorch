#!/bin/bash
cd ../src
CUDA_VISIBLE_DEVICES=2,3 python main.py --head_conv 256 --test --val_intervals 1 --gpus 2,3 --task multi_pose --load_model ../models/multi_pose_dla_3x.pth --dataset coco_hp --exp_id dla_3x --arch dla_34 --batch_size 4 --master_batch 2 --lr 1.25e-5 

# test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume
# flip test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test 
# multi scale test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5

