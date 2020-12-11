cd ../src
python demo.py --task multi_pose_whole --dataset wholebody --head_conv 24 --demo ../data/coco/debug_imgs --arch mobilev3_10 --load_model ../exp/multi_pose_whole/1211_obj600_whole_body_mobilev3_10_800_800_bs36_5e-5/model_last.pth
