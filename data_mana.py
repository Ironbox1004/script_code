#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-11 9:59
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : data_mana.py

import xml.etree.ElementTree as ET
import os
from os.path import join as pjoin
import shutil

class Data_Management():
    """
    多维子数据集管理
    """
    def __init__(self, src, save_dir, odr_name):
        self.src = src
        self.save_dir = save_dir
        self.odr_name = odr_name

    def save_file(self,file_name,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy(file_name,save_dir)

    def data_management(self):
        for category_name in os.listdir(self.src):
            annotations_path = pjoin(self.src, category_name)
            tree = ET.parse(annotations_path)
            objects = []
            for obj in tree.findall("object"):
                obj_struct = {}
                obj_struct["name"] = obj.find("name").text
                obj_struct["pose"] = obj.find("pose").text
                obj_struct["truncated"] = int(obj.find("truncated").text)
                obj_struct["difficult"] = int(obj.find("difficult").text)
                bbox = obj.find("bndbox")
                obj_struct["bbox"] = [
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text),
                    int(bbox.find("ymax").text),
                ]

                if obj_struct["name"] == self.odr_name  :
                    save_path = pjoin(self.save_dir,obj_struct["name"])
                    print(save_path)
                    self.save_file(annotations_path,save_path)
        return objects




if __name__ == '__main__':
    src = "E:\datang\YOLOX-0.2.0_DT_multi\datasets\yz2voc_multiClass\VOCdevkit\VOC_YZ20220615\Annotations"
    name = "Truck"
    save_dir = "E:\data"
    D = Data_Management(src,save_dir,name)
    data = D.data_management()
