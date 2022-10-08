#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-29 16:37
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : voc2coco.py
import json
import os
import shutil
import datetime
from PIL import Image
from tqdm import trange

root_dir = "/home/chenzhen/code/detection/datasets/union2voc_multiClass"


def voc2coco():
    # 处理coco数据集中category字段。
    # 创建一个 {类名 : id} 的字典，并保存到 总标签data 字典中。
    #['Car', 'Bus', 'Cyclist', 'Pedestrian', 'driverless_car', 'Truck', 'Tricyclist', 'Trafficcone']
    class_name_to_id = {'Car': 0, 'Bus': 1, 'Cyclist': 2, 'Pedestrian': 3, 'driverless_car': 4, 'Truck': 5, 'Tricyclist': 6, 'Trafficcone': 7}        # 改为自己的类别名称，以及对应的类别id

    # 创建coco的文件夹
    if not os.path.exists(os.path.join(root_dir, "coco")):
        os.makedirs(os.path.join(root_dir, "coco"))
        os.makedirs(os.path.join(root_dir, "coco", "annotations"))
        os.makedirs(os.path.join(root_dir, "coco", "train"))
        os.makedirs(os.path.join(root_dir, "coco", "val"))

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
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
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
    images_dir = os.path.join(root_dir, 'VOCdevkit', 'VOC_UnDt20220823', 'JPEGImages')
    images = os.listdir(images_dir)

    # 生成每个图片对应的image_id
    images_id = {}
    for idx, image_name in enumerate(images):
        images_id.update({image_name[:-4]: idx})

    # 获取训练图片
    train_img = []
    fp = open(os.path.join(root_dir, 'VOCdevkit', 'VOC_UnDt20220823', 'ImageSets', 'Main', 'train.txt'))
    for line in fp.readlines():
        list = line[:-1]
        file_name = (list.split('/')[-1]).split('.')
        if len(file_name) == 1:
            copy_name = str(file_name[0])  + ".jpg"
        else:
            copy_name = str(file_name[0]) + '.' + str(file_name[1]) + ".jpg"
        # train_img.append(i[:-1] + ".jpg")
        train_img.append(copy_name)
    # 获取训练图片的数据
    for image in train_img:
        img = Image.open(os.path.join(images_dir, image))
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=image,  # 图片的文件名带后缀
                height=img.height,
                width=img.width,
                date_captured=None,
                # id=image[:-4],
                id=images_id[image[:-4]],
            )
        )

    # 获取coco数据集train中annotations字段。
    train_xml = [i[:-4] + '.xml' for i in train_img]

    bbox_id = 0
    for xml in train_xml:
        category = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        ignore_list = []
        import xml.etree.ElementTree as ET
        tree = ET.parse(os.path.join(root_dir, 'VOCdevkit', 'VOC_UnDt20220823', 'ignore_xml_1', xml))
        root = tree.getroot()
        object = root.findall('object')
        for i in object:
            category.append(class_name_to_id[i.findall('name')[0].text])
            bndbox = i.findall('bndbox')
            ignore = i.findall('ignore')[0].text
            ignore_list.append(ignore)
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
                    area=(xmax[i] - xmin[i]) * (ymax[i] - ymin[i]),
                    bbox=[xmin[i], ymin[i], xmax[i] - xmin[i], ymax[i] - ymin[i]],
                    iscrowd=int(ignore_list[i]),
                )
            )
            bbox_id += 1
    # 生成训练集的json
    json.dump(data, open(os.path.join(root_dir, 'coco', 'annotations', 'train.json'), 'w'))

    # 获取验证图片
    val_img = []
    fp = open(os.path.join(root_dir, 'VOCdevkit', 'VOC_UnDt20220823', 'ImageSets', 'Main', 'val.txt'))
    for line in fp.readlines():
        list = line[:-1]
        file_name = (list.split('/')[-1]).split('.')
        if len(file_name) == 1:
            copy_name1 = str(file_name[0]) + ".jpg"
        else:
            copy_name1 = str(file_name[0]) + '.' + str(file_name[1]) + ".jpg"
        # val_img.append(i[:-1] + ".jpg")
        val_img.append(copy_name1)
    # 将训练的images和annotations清空，
    del data['images']
    data['images'] = []
    del data['annotations']
    data['annotations'] = []

    # 获取验证集图片的数据
    for image in val_img:
        img = Image.open(os.path.join(images_dir, image))
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=image,  # 图片的文件名带后缀
                height=img.height,
                width=img.width,
                date_captured=None,
                id=images_id[image[:-4]],   # 图片名作为id
            )
        )

    # 处理coco数据集验证集中annotations字段。
    val_xml = [i[:-4] + '.xml' for i in val_img]

    for xml in val_xml:
        category = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        ignore_list = []
        import xml.etree.ElementTree as ET
        tree = ET.parse(os.path.join(root_dir, 'VOCdevkit', 'VOC_UnDt20220823', 'ignore_xml_1', xml))
        root = tree.getroot()
        object = root.findall('object')
        for i in object:
            category.append(class_name_to_id[i.findall('name')[0].text])
            bndbox = i.findall('bndbox')
            ignore = i.findall('ignore')[0].text
            ignore_list.append(ignore)
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
                    area=(xmax[i] - xmin[i]) * (ymax[i] - ymin[i]),
                    bbox=[xmin[i], ymin[i], xmax[i] - xmin[i], ymax[i] - ymin[i]],
                    iscrowd=int(ignore_list[i]),
                )
            )
            bbox_id += 1
    # 生成验证集的json
    json.dump(data, open(os.path.join(root_dir, 'coco', 'annotations', 'val.json'), 'w'))
    print('| VOC -> COCO annotations transform finish.')
    print('Start copy images...')

    # 复制图片
    m = len(train_img)
    for i in trange(m):
        shutil.copy(os.path.join(images_dir, train_img[i]), os.path.join(root_dir, 'coco', 'train', train_img[i]))
    print('| Train images copy finish.')

    m = len(val_img)
    for i in trange(m):
        shutil.copy(os.path.join(images_dir, val_img[i]), os.path.join(root_dir, 'coco', 'val', val_img[i]))
    print('| Val images copy finish.')


if __name__ == '__main__':
    voc2coco()

