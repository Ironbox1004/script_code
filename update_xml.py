
import os
import xml.etree.ElementTree as ET
from os.path import join as pjoin
from tqdm import tqdm


src = "/home/chenzhen/code/detection/datasets/Dair_x2x/cooperative-vehicle-infrastructure-xml"

for category_name in tqdm(os.listdir(src)):
    annotations_path = pjoin(src, category_name)
    # print(annotations_path)
    tree = ET.parse(annotations_path)
    root = tree.getroot()
    objs = root.findall('object')
    xyxy = []
    for obj in objs:
        if obj.find("name").text == "Van":
            obj.find("name").text = "Car"

        elif obj.find("name").text == "Cyclist":
            obj.find("name").text = 'Cycling'

        elif obj.find("name").text == "Cyclist":
            obj.find("name").text = 'Cycling'

        elif obj.find("name").text == "Motorcyclist":
            obj.find("name").text = 'Cycling'

        elif obj.find("name").text == "Barrowlist":
            obj.find("name").text = 'Cycling'

        elif obj.find("name").text == "Trafficcone":
            obj.find("name").text = 'Special_Target'

    tree.write(annotations_path,encoding='utf-8')