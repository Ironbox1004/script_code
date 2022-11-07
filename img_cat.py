
import os

import cv2

img_path1 = '/home/chenzhen/code/detection/dt_mmdetection/output/output_bbox/'
img_path2 = '/home/chenzhen/code/detection/dt_mmdetection/output/output_bbox_40/'
img_save = "/home/chenzhen/code/detection/dt_mmdetection/output/test/"

for imgname in os.listdir(img_path1):
    img1_path = img_path1 + str(imgname)
    img1 = cv2.imread(img1_path)

    img2_path = img_path2 + str(imgname)
    img2 = cv2.imread(img2_path)

    m_h = cv2.hconcat([img2, img2])
    save_path = img_save + str(imgname)

    cv2.imwrite(save_path,m_h)