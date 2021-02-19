import shutil
import cv2

f = open('../data/wider_face/crop_lefthand/val.txt', 'r')
imgs = f.readlines()
for i in imgs:
    shutil.copy('../data/wider_face/crop_lefthand/imgs/' + i[:-1] + '.jpg', '../data/wider_face/crop_lefthand/val_imgs/' + i[:-1] + '.jpg')
    img = cv2.imread('../data/wider_face/crop_lefthand/imgs/' + i[:-1] + '.jpg')
    height, width = img.shape[0], img.shape[1]
    if height > 50 and width > 50:
        shutil.copy('../data/wider_face/crop_lefthand/imgs/' + i[:-1] + '.jpg', '../data/wider_face/crop_lefthand/easy_val_imgs/' + i[:-1] + '.jpg')
