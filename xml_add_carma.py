import os
import xml.etree.ElementTree as ET
from os.path import join as pjoin
from tqdm import tqdm


src = "/home/chenzhen/code/detection/datasets/Dair_x2x/cooperative-vehicle-infrastructure-xml-add"

for category_name in tqdm(os.listdir(src)):
    annotations_path = pjoin(src, category_name)
    # print(annotations_path)
    tree = ET.parse(annotations_path)
    root = tree.getroot()

    tree.write(annotations_path, encoding='utf-8')