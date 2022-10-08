#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-11 14:09
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : data_management.py
"""
1.获取标注文件目录
2.根据判定条件，保存满足条件的标注文件目录
"""

import xml.etree.ElementTree as ET
from os.path import join as pjoin
from lxml import etree, objectify
import os

area_poly = [[1273,89],[1492,89],[1492,279],[1273,279]]



def is_poi_in_poly(pt, poly):
    """
    判断点是否在多边形内部的 pnpoly 算法
    :param pt: 点坐标 [x,y]
    :param poly: 点多边形坐标 [[x1,y1],[x2,y2],...]
    :return: 点是否在多边形之内
    """
    nvert = len(poly)
    vertx = []
    verty = []
    testx = pt[0]
    testy = pt[1]
    for item in poly:
        vertx.append(item[0])
        verty.append(item[1])
    j = nvert - 1
    res = False
    for i in range(nvert):
        if (verty[j] - verty[i]) == 0:
            j = i
            continue
        x = (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]
        if ((verty[i] > testy) != (verty[j] > testy)) and (testx < x):
            res = not res
        j = i
    return res
def in_poly_area_dangerous(xyxy,area_poly):
    """
    检测人体是否在多边形危险区域内
    :param xyxy: 人体框的坐标
    :param img_name: 检测的图片标号，用这个来对应图片的危险区域信息
    :return: True -> 在危险区域内，False -> 不在危险区域内
    """
    # print(area_poly)
    if not area_poly:  # 为空
        return False
    # 求物体框的中点
    object_x1 = int(xyxy[0])
    object_y1 = int(xyxy[1])
    object_x2 = int(xyxy[2])
    object_y2 = int(xyxy[3])
    object_w = object_x2 - object_x1
    object_h = object_y2 - object_y1
    object_cx = object_x1 + (object_w / 2)
    object_cy = object_y1 + (object_h / 2)
    return is_poi_in_poly([object_cx, object_cy], area_poly)





def data_management(src,save_dir):
    """
    args:src标注文件目录
        :save_dir保存的标注文件目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for category_name in os.listdir(src):

        annotations_path = pjoin(src, category_name)

        tree = ET.parse(annotations_path)
        folder = tree.find("folder").text
        filename = tree.find("filename").text
        annopath = save_dir + str(filename[:-3]) + "xml"

        database,  width, height, depth = get_info(annotations_path) #原标注文件信息获取
        objects = get_obj(annotations_path)     #原标注文件信息获取

        #标注信息保存 头文件信息
        E = objectify.ElementMaker(annotate=False)

        anno_tree = E.annotation(
            E.folder(folder),
            E.filename(filename),
            E.source(
                E.database(database),
                E.annotation("DT"),
            ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(depth)
            )
        )
        #标注信息保存，标注信息
        xyxy = []
        for obj in objects:
            if obj["name"] == "Cyclist":
                xyxy = [int(obj["bbox"][0]), int(obj["bbox"][1]), int(obj["bbox"][2]), int(obj["bbox"][3])]
                area = (int(obj["bbox"][2]) - int(obj["bbox"][0])) * (int(obj["bbox"][3]) - int(obj["bbox"][1]))
                if in_poly_area_dangerous(xyxy, area_poly) == True and area > 4000:
                        E2 = objectify.ElementMaker(annotate=False)
                        anno_tree2 = E2.object(
                            E.name(obj["name"]),
                            E.pose(obj["pose"]),
                            E.truncated(obj["truncated"]),
                            E.difficult(obj["difficult"]),
                            E.ignore("1"),
                            E.bndbox(
                                E.xmin(obj["bbox"][0]),
                                E.ymin(obj["bbox"][1]),
                                E.xmax(obj["bbox"][2]),
                                E.ymax(obj["bbox"][3])
                            ))

                else:
                    E2 = objectify.ElementMaker(annotate=False)
                    anno_tree2 = E2.object(
                        E.name(obj["name"]),
                        E.pose(obj["pose"]),
                        E.truncated(obj["truncated"]),
                        E.difficult(obj["difficult"]),
                        E.ignore("0"),
                        E.bndbox(
                            E.xmin(obj["bbox"][0]),
                            E.ymin(obj["bbox"][1]),
                            E.xmax(obj["bbox"][2]),
                            E.ymax(obj["bbox"][3])
                        ))

            else:
                E2 = objectify.ElementMaker(annotate=False)
                anno_tree2 = E2.object(
                    E.name(obj["name"]),
                    E.pose(obj["pose"]),
                    E.truncated(obj["truncated"]),
                    E.difficult(obj["difficult"]),
                    E.ignore("0"),
                    E.bndbox(
                        E.xmin(obj["bbox"][0]),
                        E.ymin(obj["bbox"][1]),
                        E.xmax(obj["bbox"][2]),
                        E.ymax(obj["bbox"][3])
                    )
                )
            anno_tree.append(anno_tree2)
            etree.ElementTree(anno_tree).write(annopath, pretty_print=True)

def get_obj(annotations_path):
    """
    标注信息获取
    """
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

def get_info(annotations_path):
    """
    标注头文件信息获取
    """
    tree = ET.parse(annotations_path)

    for info in tree.findall("source"):
        database = info.find("database").text
        # annotation = info.find("annotation").text

    for info in tree.findall("size"):
        width = int(info.find("width").text)
        height = int(info.find("height").text)
        depth = int(info.find("depth").text)

    return database,  width, height, depth


if __name__ == '__main__':
    src = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/Annotations"
    save_dir = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ignore_xml/"
    # annotations_path = "E:\datang\YOLOX-0.2.0_DT_multi\datasets\yz2voc_multiClass\VOCdevkit\VOC_YZ20220615\Annotations\\1647309767.871504307.xml"
    # database, annotation, width, height, depth = get_info(annotations_path)
    # print(database,annotation,width,height,depth)
    data_management(src,save_dir)
