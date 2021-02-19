cd ../src
# python test.py --task multi_pose --dataset coco_hp --exp_id mobilev3_10 --keep_res --load_model ../exp/multi_pose/0214_coco_hp_mobilev3_10_pan_800_800/model_last.pth

python test.py --task multi_pose --dataset coco_hp --head_conv 24 --arch mobilev3_10 --exp_id mobilev3_10 --keep_res --load_model ../exp/multi_pose/0214_coco_hp_with_keypoints_mobilev3_10_pan_800_800/model_last.pth

# python test.py --task multi_pose --dataset coco_hp --head_conv 24 --arch mobilev3_10 --exp_id mobilev3_10 --keep_res --load_model ../exp/multi_pose/0214_coco_hp_mobilev3_10_pan_800_800/model_last.pth

