#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-11 14:32
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : use.py
import math
from os.path import join as pjoin
import os
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from pandas import DataFrame

def cal_area(bbox):
    x_min = int(bbox.find("xmin").text)
    x_max = int(bbox.find("xmax").text)
    y_min = int(bbox.find("ymin").text)
    y_max = int(bbox.find("ymax").text)
    area = (x_max - x_min) * (y_max - y_min)
    return area

def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
count = 0
bus_area = []
path = "/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221115/Annotations"
for category_name in tqdm(os.listdir(path)):
    annotations_path = pjoin(path, category_name)
    tree = ET.parse(annotations_path)

    for obj in tree.findall("object"):
        # count = count + 1
        # bbox = obj.find("bndbox")
        # area = cal_area(bbox)
        # bus_area.append(math.sqrt(area))
        if obj.find("name").text == "Unmanned_riding":
            count = count + 1
            bbox = obj.find("bndbox")
            area = cal_area(bbox)
            # print(area)
            bus_area.append(math.sqrt(area))
print(count)

# x = range(len(bus_area))
# plt.plot(x,bus_area)
# plt.show()

n, bins, patches = plt.hist(x=bus_area, bins="auto", color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('area')
plt.ylabel('Frequency')
plt.title('area--frequency')
maxfreq = n.max()
# 设置y轴的上限
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()