import os
import xml.etree.ElementTree as ET
from os.path import join as pjoin
from tqdm import tqdm


src = "/home/chenzhen/code/detection/datasets/repo3d/val/xml_label"
classes_name = []
for category_name in tqdm(os.listdir(src)):
    annotations_path = pjoin(src, category_name)
    # print(annotations_path)
    tree = ET.parse(annotations_path)
    root = tree.getroot()
    objs = root.findall('object')
    for obj in objs:
        if obj.find("name").text not in classes_name:
            classes_name.append(obj.find("name").text)

print(classes_name)