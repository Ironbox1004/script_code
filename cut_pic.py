import xml.etree.ElementTree as ET
import os
from os.path import join as pjoin
import cv2
from tqdm import tqdm

root_path = "/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221111"

xml_path = pjoin(root_path, 'Annotations')
pic_path = pjoin(root_path, 'JPEGImages')
save_path = pjoin(root_path, "save/")

num_list = ['ride','special target','car','truck','pedestrian','Tricycle','big special vehicle','engineering vehicler',
            'bus','dock','Small special vehicle','Jingdong car','movable','other movable objects','Other special vehicles',
            'anlmal']


for category_name in tqdm(os.listdir(xml_path)):
    list = category_name
    name = list.split('.')
    pic_name = str(name[0]+'.'+name[1]+'.jpg')
    annotations_path = pjoin(xml_path, category_name)
    tree = ET.parse(annotations_path)
    objects = []
    img_path = pjoin(pic_path, pic_name)
    img = cv2.imread(img_path)
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        if obj_struct["name"] == 'other movable objects':
            bbox_list = []
            bbox = obj.find("bndbox")
            bbox_list = [
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text),
                    int(bbox.find("ymax").text),
                ]
                # print(obj_struct["bbox"])
            cropped = img[bbox_list[1]:bbox_list[3], bbox_list[0]:bbox_list[2]]  # 裁剪坐标为[y0:y1, x0:x1]
            import uuid
            save_path_name = save_path + str(uuid.uuid1()) + str(obj_struct["name"]) + ".jpg"
            try:
                cv2.imwrite(save_path_name, cropped)
            except:
                pass