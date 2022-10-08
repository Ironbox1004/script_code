
from os.path import join as pjoin
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ignore_xml_1"

name_list = []


#1.判断obj是不是有需要判断的属性
# for category_name in tqdm(os.listdir(path)):
#     annotations_path = pjoin(path, category_name)
#     updateTree = ET.parse(annotations_path)
#     tree = ET.parse(annotations_path)
#     root = tree.getroot()
#     for obj in root.iter("ignore"):
#         if obj.text not in name_list:
#             name_list.append(obj.text)
# print(name_list)

ann_list = []

#2.获取根据属性的文件名
for category_name in tqdm(os.listdir(path)):
    annotations_path = pjoin(path, category_name)
    updateTree = ET.parse(annotations_path)
    tree = ET.parse(annotations_path)
    root = tree.getroot()
    for obj in root.iter("ignore"):
        if int(obj.text) == 1 :
            ann_list.append(category_name)
print(ann_list)