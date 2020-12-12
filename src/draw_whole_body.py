import sys
import os
import scipy.io as sio
import cv2
from scipy.io import loadmat

path = os.path.dirname(__file__)
CENTERNET_PATH = os.path.join(path,'../src/lib')
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts_pose import opts
# from opts import opts
# from opts2 import opts
from datasets.dataset_factory import get_dataset
import numpy as np

def test_img(MODEL_PATH):
    debug = 1            # draw and show the result image
    TASK = 'multi_pose'  
    input_h, intput_w = 800, 800
    opt = opts().init('--task {} --load_model {} --debug {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, intput_w, input_h).split(' '))

    detector = detector_factory[opt.task](opt)

    img = '../readme/000388.jpg'
    ret = detector.run(img)['results']


# def get_gt_boxes(gt_dir):
#     """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

#     gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
#     hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
#     medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
#     easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

#     facebox_list = gt_mat['face_bbx_list']
#     event_list = gt_mat['event_list']
#     file_list = gt_mat['file_list']

#     hard_gt_list = hard_mat['gt_list']
#     medium_gt_list = medium_mat['gt_list']
#     easy_gt_list = easy_mat['gt_list']

#     return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def test_vedio(model_path, vedio_path=None):
    debug = -1            # return the result image with draw
    TASK = 'multi_pose'  
    vis_thresh = 0.45
    input_h, intput_w = 800, 800
    opt = opts().init('--task {} --load_model {} --debug {} --input_h {} --input_w {} --vis_thresh {}'.format(
        TASK, MODEL_PATH, debug, intput_w, input_h, vis_thresh).split(' '))
    detector = detector_factory[opt.task](opt)

    vedio = vedio_path if vedio_path else 0
    cap = cv2.VideoCapture(vedio)
    while cap.isOpened():
        det = cap.grab()
        if det:
            flag, frame = cap.retrieve()
            res = detector.run(frame)
            cv2.imwrite('../outputs/', res['plot_img'])
            # cv2.imshow('face detect', res['plot_img'])

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_wider_Face(model_path):
    Path = '../data/widerface/retinaface_gt_v1.1/val/images'
    wider_face_mat = sio.loadmat('../data/widerface/wider_face_split/wider_face_val.mat')
    event_list = wider_face_mat['event_list']
    file_list = wider_face_mat['file_list']

    save_path = '../output/1212_obj600_face_hp_mobilev3_10_800_800_bs8_5e-4_ebest-1/'
    #save_path = '../output/1119_noattsmall_obj314_face_hp_mobilev3_10_800_800_bs12_5e-4_K600_ebest-1/'


    # save_path = '../output/1105_pre_face_hp_mobilev2_10_800_800_sig_bs12_5e-5_e180/'
    # save_path = '../output/1106_s_face_hp_mobilev2_10_800_800_bs12_5e-4_e90/' 
    # save_path = '../output/ctdet_opt2_face_hp_mobilev2_5_160_0.55/'
    # save_path = '../output/multi_pose_mobilev2_5_130_0.2/'
    debug = 1            # return the detect result without show
    threshold = 0.05 #0.05
    # TASK = 'multi_pose'  
    input_h, input_w = 800, 800
    # input_h, intput_w = 640, 640
    # opt = opts().init('--task ctdet --arch dla_34 --dataset facehp --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
    #      MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))
    
    # opt = opts().init('--task ctdet --test_scales 1.0 --arch mobilev2_5 --dataset face --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
    #      MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))

    # opt = opts().init('--task multi_pose --test_scales 1.0 --arch dla_34 --dataset facehp --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
    #      MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))
    
    # opt = opts().init('--task ctdet --ltrb --arch mobilev2_5 --dataset facehp --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
    #      MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))

    # --test_scales 0.5,0.75,1,1.25,1.5
    # opt = opts().init('--task multi_pose --arch mobilev2_10 --dataset facehp --exp_id {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
    #      save_path, MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))

    # opt = opts().init('--task multi_pose --arch shufflenetv2_10 --dataset facehp --exp_id {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
    #      save_path, MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))

    # opt = opts().init('--task multi_pose --arch mobilev3_10 --head_conv 24 --K 600 --dataset facehp --exp_id {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
    #      save_path, MODEL_PATH, debug, threshold, input_h, input_w).split(' '))

    opt = opts().init('--task multi_pose_whole --arch mobilev3_10 --head_conv 24 --K 600 --dataset wholebody --exp_id {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
    save_path, MODEL_PATH, debug, threshold, input_h, input_w).split(' '))
    
    detector = detector_factory[opt.task](opt)

    for index, event in enumerate(event_list):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)
        for num, file in enumerate(file_list_item):
            im_name = file[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            img_path = os.path.join(Path, zip_name)
            res = detector.run(img_path)

            dets = res['results']
            img_np = np.transpose(np.array(res['image'].cpu()), (1, 2, 0)) * 255 
            hm_pseudo = res['hm'][0].cpu()
            hm_vis = 255 - (hm_pseudo[0] - 0.5) * 255 * 2 
            hm_vis = np.clip(hm_vis, 0, 255)
            hm_vis = np.array(hm_vis, dtype=np.uint8)
            hm_vis = np.expand_dims(hm_vis, axis=-1)
            hm_vis = np.repeat(hm_vis, 3, axis=-1)
            # import pdb; pdb.set_trace()
            hm_vis = cv2.resize(hm_vis, (input_w, input_h)) 
            # [:, :, 0]
            # colored_hm = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
            colored_hm = hm_vis
            masked_image =  colored_hm * 0.9 + img_np * 0.1
            # import pdb; pdb.set_trace()
            
            img_name = '../evaluate/heatmap-hand/' + im_dir 

            if not os.path.exists(img_name):
                os.mkdir(img_name)

            cv2.imwrite('../evaluate/heatmap-hand/' + zip_name, masked_image)
            cv2.imwrite('../evaluate/heatmap-hand/' + '%s/pure_%s.jpg' % (im_dir, im_name) , colored_hm)

            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            for b in dets[1]:
                x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            f.close()

            print('event:%d num:%d' % (index + 1, num + 1))


def test_whole_body(model_path):
    Path = '../data/coco/hand_val'

    save_path = '../output/1212_obj600_whole_body_mobilev3_10_800_800_bs36_5e-4_e-30/'
    debug = 1            # return the detect result without show
    threshold = 0.01
     
    input_h, input_w = 800, 800
    
    opt = opts().init('--task multi_pose_whole --arch mobilev3_10 --head_conv 24 --K 600 --dataset wholebody --exp_id {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
    save_path, MODEL_PATH, debug, threshold, input_h, input_w).split(' '))
    
    detector = detector_factory[opt.task](opt)

    for im_name in os.listdir(Path):
        img_path = '%s/%s' % (Path, im_name)
        print(img_path)
        res = detector.run(img_path)
        dets = res['results']
        img_np = np.transpose(np.array(res['image'].cpu()), (1, 2, 0)) * 255 
        hm_pseudo = res['hm'][0].cpu()
        # import pdb; pdb.set_trace()
        hm_vis = 255 - (hm_pseudo[0]-0.5) * 255 * 2
        hm_vis = np.clip(hm_vis, 0, 255)
        hm_vis = np.array(hm_vis, dtype=np.uint8)
        hm_vis = np.expand_dims(hm_vis, axis=-1)
        hm_vis = np.repeat(hm_vis, 3, axis=-1)
        # import pdb; pdb.set_trace()
        hm_vis = cv2.resize(hm_vis, (input_w, input_h)) 
        # [:, :, 0]
        # colored_hm = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        colored_hm = hm_vis
        masked_image =  colored_hm * 0.9 + img_np * 0.1
        # import pdb; pdb.set_trace()
        
        img_dir = '../evaluate/heatmap-hand' 

        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        cv2.imwrite('../evaluate/heatmap-hand/' + im_name, masked_image)
        cv2.imwrite('../evaluate/heatmap-hand/' + 'pure_%s' % (im_name) , colored_hm)

        # f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
        # f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
        # f.write('{:d}\n'.format(len(dets)))
        # for b in dets[1]:
        #     x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
        #     f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
        # f.close()

        # print('event:%d num:%d' % (index + 1, num + 1))


if __name__ == '__main__':
    # MODEL_PATH = '../exp/multi_pose/1106_s_face_hp_mobilev2_10_800_800_bs12_5e-4/model_90.pth'
    # MODEL_PATH = '../exp/multi_pose/1105_pre_face_hp_mobilev2_10_800_800_sig_bs12_5e-5/model_180.pth'

    #MODEL_PATH = '../exp/multi_pose/1119_noattsmall_obj314_face_hp_mobilev3_10_800_800_bs12_5e-4/model_best.pth'
    MODEL_PATH = '../exp/multi_pose_whole/1212_obj600_whole_body_mobilev3_10_800_800_bs36_5e-4/model_30.pth'

    # MODEL_PATH = '../exp/multi_pose/1104_nopre_face_hp_mobilev2_10_800_800_sig_bs12_5e-4/model_140.pth'
    # MODEL_PATH = '../exp/ctdet/1002_opt2_face_hp_mobilev2_5/model_160.psth'
    # MODEL_PATH = '../exp/multi_pose/mobilev2_10/model_best.pth'
    # MODEL_PATH = './pretrained/centerface_best.pth'
    # test_img(MODEL_PATH)
    # test_vedio(MODEL_PATH)
    test_whole_body(MODEL_PATH)
