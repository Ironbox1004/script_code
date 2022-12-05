from os.path import join as pjoin
import os
import xml.etree.ElementTree as ET
import numpy as np
np.set_printoptions(precision=4, suppress=True)

xml_path = "/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221115/Annotations/"
# text_data = "/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221115/ImageSets/Main/train.txt"
text_data = "/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221115/ImageSets/Main/test.txt"

classes_list = [
        "Car",
        "Bus",
        "Cycling",
        "Pedestrian",
        "driverless_Car",
        "Truck",
        "Animal",
        "Obstacle",
        "Special_Target",
        "Other_Objects",
        "Unmanned_riding"
    ]
classes_matrix = np.zeros(shape=(len(classes_list), len(classes_list)))

with open (text_data, 'r') as f:
    for line in f :
        list = line[:-1]
        annotations_path = xml_path + list + '.xml'
        updateTree = ET.parse(annotations_path)
        tree = ET.parse(annotations_path)
        root = tree.getroot()
        for obj in root.iter("name"):
            index = classes_list.index(obj.text)
            classes_matrix[index, index] += 1

classes_nums = classes_matrix.diagonal()
print(classes_nums)