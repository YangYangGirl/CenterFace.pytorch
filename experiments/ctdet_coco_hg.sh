#!/bin/bash
cd ../src
CUDA_VISIBLE_DEVICES=2,3 python main.py  --gpus 2,3 --task ctdet --dataset coco --exp_id coco_hg --arch hourglass --batch_size 8 --master_batch 4 --lr 1.25e-4 --load_model ../models/ctdet_coco_hg.pth

# test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume
# flip test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test 
# multi scale test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5

