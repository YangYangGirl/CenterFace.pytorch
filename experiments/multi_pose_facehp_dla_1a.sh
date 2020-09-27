#!/bin/bash
cd ../src
python main.py --arch dla_34 --dataset facehp --load_model ../pretrained/ctdet_coco_dla_1x.pth --exp_id face_hp_dla --batch_size 8 --lr 5e-4   --gpus 5 --num_workers 4 --num_epochs 140 --lr_step 90,120
