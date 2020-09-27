#!/bin/bash
cd ../src
python main.py --arch mobilev2_10  --dataset facehp --exp_id face_hp_mobilev2_10 --batch_size 8 --master_batch 8 --lr 5e-4  --gpus 4,5 --num_workers 4 --num_epochs 140 --lr_step 90,120
