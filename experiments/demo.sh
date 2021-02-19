cd ../src
# python demo.py --task ctdet --input_h 800 --input_w 800 --dataset wholebody --head_conv 24 --demo ../data/coco/hand_val --arch mobilev3_10 --load_model ../exp/ctdet/1230_normal_radius_ctdet_mobilev3_10_fpn_only_box_obj600_800_800_bs16_5e-4/model_best.pth
# python demo.py --task multi_pose_crop --dataset cropbody --head_conv 24 --input_h 128 --input_w 128 --demo ../data/wider_face/crop_lefthand/debug_img --arch mobilev3_10 --load_model ../exp/multi_pose_crop/0202_mse_adam_mobilev3_10_obj600_whole_body_crop_128_128_bs2_5e-4/model_last.pth
# python demo.py --task multi_pose_crop --dataset cropbody --head_conv 24 --input_h 128 --input_w 128 --demo ../data/wider_face/crop_lefthand/debug_img --arch hourglass --load_model ../exp/multi_pose_crop/0204_mse_adam_hg_3x_obj600_whole_body_crop_128_128_bs16_5e-4/model_last.pth

# python demo.py --task multi_pose_crop --dataset cropbody --head_conv 24 --input_h 128 --input_w 128 --demo ../data/wider_face/crop_lefthand/easy_val_imgs --arch hourglass --load_model ../exp/multi_pose_crop/0204_nomse_adam_hg_3x_obj600_whole_body_crop_128_128_bs16_5e-4/model_last.pth

# python demo.py --task multi_pose --dataset coco_hp --head_conv 24 --input_h 800 --input_w 800 --demo ../data/coco/test2017 --arch mobilev3_10 --load_model ../exp/multi_pose/0214_coco_hp_mobilev3_10_pan_800_800/model_last.pth
python demo.py --task multi_pose --dataset coco_hp --head_conv 24 --input_h 800 --input_w 800 --demo ../data/coco/body_val --arch mobilev3_10 --load_model ../exp/multi_pose/0214_coco_hp_with_keypoints_mobilev3_10_pan_800_800/model_last.pth

# python demo.py --task multi_pose_crop --dataset cropbody --head_conv 24 --input_h 128 --input_w 128 --demo ../data/wider_face/crop_lefthand/easy_val_imgs --arch mobilev3_10 --load_model ../exp/multi_pose_crop/0204_nomse_adam_mobilev3_10_obj600_whole_body_crop_128_128_bs16_5e-4/model_150.pth
