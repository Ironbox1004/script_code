#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-09-01 9:38
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : xml2txt.py
import xml.etree.ElementTree as ET
import os
from glob import glob


# VOC格式的xml转YOLO格式的txt
def convert(size, box):
    '''
    size = (w, h)
    box = (xmin, xmax, ymin, ymax)
    '''
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    if x > 1:
        print(x)
    return x, y, w, h


# 单个标注文件的转换
def convert_annotation(image_id):
    res = ""
    in_file = open('{}/{}.xml'.format(xml_path, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes :
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        ignore = obj.find('ignore').text
        clu1 = 0
        clu2 = 1
        clu3 = 2
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        res += "{} {} {} {} {} {} {} {} {}\n".format(cls_id, bb[0], bb[1], bb[2], bb[3],ignore, clu1,clu2,clu3,)
    return res


# 写文件
def write_file(fname, content):
    with open(fname, 'w') as f:
        f.write(content)


# 批量转换
def generate_label_file():
    # 获取所有xml文件
    xmls = glob("{}/*.xml".format(xml_path))
    xml_ids = [v.split(os.sep)[-1].split('.xml')[0] for v in xmls]
    for xml_id in xml_ids:
        content = convert_annotation(xml_id)
        if content != "":
            lable_file = "{}/{}.txt".format(txt_path, xml_id)
            write_file(lable_file, content)


if __name__ == '__main__':
    # xml文件路径、txt文件保存路径
    xml_path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ignore_xml"
    txt_path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ignore_text"

    # 类别列表
    classes = ['Car', 'Bus', 'Cyclist', 'Pedestrian', 'driverless_car', 'Truck', 'Tricyclist', 'Trafficcone']

    # 执行转换
    generate_label_file()
