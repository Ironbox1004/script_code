#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-29 16:37
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : voc2coco.py
import json
import os
import xml.etree.ElementTree as ET
import shutil
import datetime
from PIL import Image
from tqdm import trange

root_dir = "/home/chenzhen/code/detection/datasets/Dair_x2x"


def voc2coco():
    class_name_to_id = {
        "Car": 0, "Bus": 1, "Cycling": 2, "Pedestrian": 3, "driverless_Car": 4, "Truck": 5, "Animal": 6, "Obstacle": 7, "Special_Target": 8, "Other_Objects": 9, "Unmanned_riding": 10
    }
    # 创建coco的文件夹
    if not os.path.exists(os.path.join(root_dir, "coco_single")):
        os.makedirs(os.path.join(root_dir, "coco_single"))
        os.makedirs(os.path.join(root_dir, "coco_single", "annotations"))
        os.makedirs(os.path.join(root_dir, "coco_single", "train"))
        os.makedirs(os.path.join(root_dir, "coco_single", "val"))

    # 创建 总标签data
    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, file_name,url, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
             #'segmentation, area, iscrowd, image_id, bbox, category_id, id'
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    for name, id in class_name_to_id.items():
        data["categories"].append(
            dict(supercategory=None, id=id, name=name, )
        )

    # 处理coco数据集train中images字段。
    images_dir = os.path.join(root_dir,  'single-infrastructure-side-image_9807202513588224', 'single-infrastructure-side-image')
    images = os.listdir(images_dir)

    # 生成每个图片对应的image_id
    images_id = {}
    for idx, image_name in enumerate(images):
        images_id.update({image_name[:-4]: idx})

    # 获取训练图片
    train_img = []
    fp = os.path.join(root_dir,  'single-infrastructure-side-image_9807202513588224', 'single-infrastructure-side-image')
    for line in os.listdir(fp):
        train_img.append(line)
    # 获取训练图片的数据
    bbox_id = 0
    for image in train_img:
        img = Image.open(os.path.join(images_dir, image))
        category = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        occlude_list = []
        truncation_factor_list = []
        xml = image[:-4] + '.xml'
        tree = ET.parse(os.path.join(root_dir, 'single-infrastructure-side-xml', xml))
        root = tree.getroot()
        size = root.findall('size')
        for element in size:
            date_captured = element.find('date_captured').text
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=image,  # 图片的文件名带后缀
                height=img.height,
                width=img.width,
                date_captured=str(date_captured),
                # id=image[:-4],
                id=images_id[image[:-4]],
            )
        )
        object = root.findall('object')
        for i in object:
            category.append(class_name_to_id[i.findall('name')[0].text])
            bndbox = i.findall('bndbox')
            occlude = i.findall('occ')[0].text
            occlude_list.append(occlude)
            truncation_factor = i.findall('direct')[0].text
            truncation_factor_list.append(truncation_factor)

            for j in bndbox:
                xmin.append(float(j.findall('xmin')[0].text))
                ymin.append(float(j.findall('ymin')[0].text))
                xmax.append(float(j.findall('xmax')[0].text))
                ymax.append(float(j.findall('ymax')[0].text))
        for i in range(len(category)):
            data["annotations"].append(
                dict(
                    id=bbox_id,
                    image_id=images_id[xml[:-4]],
                    category_id=category[i],
                    # visibility=int(visibility_list[i]),
                    occ=int(occlude_list[i]),
                    direct=int(truncation_factor_list[i]),
                    iscrowd=int(0),
                    area=(xmax[i] - xmin[i]) * (ymax[i] - ymin[i]),
                    bbox=[xmin[i], ymin[i], xmax[i] - xmin[i], ymax[i] - ymin[i]]
                )
            )
            bbox_id += 1
    # 生成训练集的json
    json.dump(data, open(os.path.join(root_dir, 'coco_single', 'annotations', 'train.json'), 'w'))




if __name__ == '__main__':
    voc2coco()

