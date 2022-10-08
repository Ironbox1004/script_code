#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-23 10:45
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : process_xml.py

from os.path import join as pjoin
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

path = r"E:\detection\datasets\union2voc_multiClass\VOCdevkit\VOC_UnDt20220823\process"
num_list = ['Car', 'Bus', 'Cyclist', 'Pedestrian', 'driverless_car', 'Truck', 'Tricyclist', 'Trafficcone']
car_num, bus_num,cyclist_num,pede_num,driver_num,truck_num,tricyclist_num,trafficone_num,barrowist_num = 0,0,0,0,0,0,0,0,0
for category_name in tqdm(os.listdir(path)):
    annotations_path = pjoin(path, category_name)
    updateTree = ET.parse(annotations_path)
    tree = ET.parse(annotations_path)
    root = tree.getroot()
    for obj in root.iter("name"):
        if obj.text == "Car":
            car_num = car_num + 1
        elif obj.text == "Bus":
            bus_num = bus_num + 1
        elif obj.text == "Cyclist":
            cyclist_num = cyclist_num + 1
        elif obj.text == "Pedestrian":
            pede_num = pede_num + 1
        elif obj.text == "driverless_car":
            driver_num = driver_num + 1
        elif obj.text == "Truck":
            truck_num = truck_num + 1
        elif obj.text == "Tricyclist":
            tricyclist_num = tricyclist_num + 1
        elif obj.text == "Trafficcone":
            trafficone_num = trafficone_num + 1

print(num_list)
print(car_num,bus_num,cyclist_num,pede_num,driver_num,truck_num,tricyclist_num,trafficone_num)