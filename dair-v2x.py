#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-22 14:58
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : dair-v2x.py

import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import os
import time
import json


def ConvertVOCXml(file_path="", file_name=""):
    xml_file = open((r'E:\download\DAIR-V2X-I\single-infrastructure-side\label\result\\' + file_name[:-5] + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>'+file_name[:-5]+'.jpg'+'</filename>\n')

    xml_file.write('    <source>\n')
    xml_file.write('        <database>' + 'DAIR-V2X' + '</database>\n')
    xml_file.write('        <annotation>' + 'DT' + '</annotation>\n')
    xml_file.write('    </source>\n')

    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + '1920' + '</width>\n')
    xml_file.write('        <height>' + '1080' + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    f = open((r'E:\download\DAIR-V2X-I\single-infrastructure-side\label\camera_test\\' + file_name))

    content = json.load(f)
    for i in range(len(content)):
        gt_label = content[i]['type']
        xmin = content[i]['2d_box']['xmin']
        ymin = content[i]['2d_box']['ymin']
        xmax = content[i]['2d_box']['xmax']
        ymax = content[i]['2d_box']['ymax']
        xmin, ymin, xmax, ymax = int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))

        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(gt_label) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(xmin) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(ymin) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(xmax) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(ymax) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')
    xml_file.close()


if __name__ == "__main__":
    basePath = r'E:\download\DAIR-V2X-I\single-infrastructure-side\label\camera_test\\'  # dair-v2x数据集的json文件放置位置
    totaljson = os.listdir(basePath)
    totaljson.sort()
    total_num = 0
    flag = False
    print("正在转换")
    saveBasePath = r'E:\download\DAIR-V2X-I\single-infrastructure-side\label\result\\'  # voc格式的xml文件放置位置
    if os.path.exists(saveBasePath) == False:  # 判断文件夹是否存在
        os.makedirs(saveBasePath)

    # ConvertVOCXml(file_path="samplexml",file_name="000009.xml")
    # Start time
    start = time.time()
    for json1 in totaljson:
        file_name = os.path.join(basePath, json1)
        print(file_name)

        ConvertVOCXml(file_path=saveBasePath, file_name=json1)

    # End time
    end = time.time()
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))
