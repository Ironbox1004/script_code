import shutil

import imageio
import numpy as np
import os
import cv2


#1.
# f = open(r"D:\dt_detection\datasets\union2voc_multiClass\yolo_dataset\val.txt")
# xml_path = r"D:\dt_detection\datasets\union2voc_multiClass\VOCdevkit\VOC_UnDt20220823\JPEGImages"
# copy_xml_path = r"D:\dt_detection\datasets\test_img"
#
# with open (r"D:\dt_detection\datasets\union2voc_multiClass\yolo_dataset\val.txt", 'r') as f:
#     for line in f :
#         list = line[:-1]
#         file_name = (list.split('\\')[-1]).split('.')
#         if len(file_name) == 2:
#             copy_name = xml_path + '\\' + str(file_name[0]) + '.jpg'
#         else:
#             copy_name = xml_path + '\\' + str(file_name[0]) + '.' + str(file_name[1]) + '.jpg'
#         # copy_name = xml_path + '\\' + file_name +'.xml'
#         shutil.copy(copy_name, copy_xml_path)



#2.
# xml_path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ignore_xml/"
# path_path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/JPEGImages/"
#
# xml_copy_path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/copy_data/xml"
# pic_copy_path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/copy_data/pic"
#
# copy_list = ['1648622287.209941387.xml', '1647339200.212722540.xml', '1648622342.537121534.xml',
#              '1647339567.566006660.xml', '1648623462.924392223.xml', '1648622442.629086971.xml',
#              '1647339019.071483612.xml', '1647339183.489666700.xml', '1648621901.500053167.xml',
#              '1648623442.957228661.xml', '1647339018.559798717.xml', '1647339428.390323877.xml',
#              '1648623003.260711670.xml', '1648623600.624649286.xml', '1648622339.790396929.xml',
#              '1647334452.574007273.xml', '1648621902.503090143.xml', '1647338427.040148020.xml']
#
#
# for name in copy_list:
#     name_spl = name.split(".")[0:2]
#     copy_name = xml_path + str(name_spl[0]) + "." + str(name_spl[1]) + ".xml"
#     shutil.copy(copy_name, xml_copy_path)

# list_1 = []
# lis = os.listdir("/home/chenzhen/code/detection/datasets/union2voc_multiClass/copy_data/xml")
# for name in lis:
#     list_1.append(name)
# print(list_1)

import shutil
import os



img_path = '/home/chenzhen/code/detection/dt_mmdetection/output/output_bbox_2399_imgs'
xml_path = '/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ignore_xml/'
save_xml_path = '/home/chenzhen/code/detection/dt_mmdetection/output/xml'

for img_list in os.listdir(img_path):
    img_name = img_list.split('.')
    if len(img_name) == 2:
        copy_name = xml_path + str(img_name[0])  + '.xml'
        shutil.copy(copy_name, save_xml_path)
    else:
        copy_name = xml_path + str(img_name[0]) + '.' + str(img_name[1]) + '.xml'
        shutil.copy(copy_name, save_xml_path)
