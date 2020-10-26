cd ../src
CUDA_VISIBLE_DEVICES=1,3 python main.py --arch dla_34  --load_model ../pretrained/ctdet_coco_dla_1x.pth --task ctdet --dataset face --exp_id 1007_opt2_face_dla_34 --batch_size 16 --master_batch 8 --lr 10e-4   --gpus 1,3 --num_workers 4 --num_epochs 200 --lr_step 45,60

