#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-11 11:06
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : test.py


import xml.etree.ElementTree as ET
import os
from os.path import join as pjoin
import shutil
from lxml import etree, objectify


def data_management(annotations_path):

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
        objects.append(obj_struct)
    return objects

def save_annotations(objects):
    annopath = "E:\data\Truck.xml"
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC_YZ20220615'),
        E.filename("1647309767.871504307.jpg"),
        E.source(
            E.database('YZ20220615'),
            E.annotation('DT'),
        ),
        E.size(
            E.width(1920),
            E.height(1080),
            E.depth(3)
        )
    )
    for obj in objects:
        if obj["name"] == "Truck":
            E2 = objectify.ElementMaker(annotate=False)
            anno_tree2 = E2.object(
                E.name(obj["name"]),
                E.pose(),
                E.truncated("0"),
                E.difficult(0),
                E.bndbox(
                    E.xmin(obj["bbox"][0]),
                    E.ymin(obj["bbox"][1]),
                    E.xmax(obj["bbox"][2]),
                    E.ymax(obj["bbox"][3])
                )
            )
            anno_tree.append(anno_tree2)
        etree.ElementTree(anno_tree).write(annopath, pretty_print=True)





if __name__ == '__main__':
    src = "E:\data\Truck\\1647310326.879065752.xml"
    d = data_management(src)
    # print(d)
    save_annotations(d)