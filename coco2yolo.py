#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-31 17:44
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : coco2yolo.py
# coco是x1,y1,w,h，yolo是x,y,w,h。 x1,y1是左上角坐标，x,y是中心坐标
import os
import shutil

from pycocotools.coco import COCO
from tqdm import trange

root_dir = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/"


# 将coco的bbox转换为yolo的bbox
def cocobbox2yolobbox(coco_box):
    box = coco_box
    dw = 1. / 1920
    dh = 1. / 1080
    x = (float(box[0]) + float(box[1])) / 2.0
    y = (float(box[2]) + float(box[3])) / 2.0
    w = float(box[1]) - float(box[0])
    h = float(box[3]) - float(box[2])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def coco2yolo(dataset_type, json_fp, origin_imgs_dir, save_dir):
    imgs_dir = os.path.join(save_dir, 'images')
    labels_dir = os.path.join(save_dir, 'labels')
    ImageSets_dir = os.path.join(save_dir, 'ImageSets')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(imgs_dir)
        os.mkdir(labels_dir)
        os.mkdir(ImageSets_dir)

    text_data_fp = os.path.join(ImageSets_dir, dataset_type + '.txt')
    text_data = []

    coco = COCO(json_fp)
    imgs = coco.imgs
    img_ids = coco.getImgIds()

    m = len(img_ids)
    for i in trange(m):
        img_id = img_ids[i]
        filename = imgs[img_id]['file_name']
        text_data.append(os.path.join(imgs_dir, filename))
        # text_data.append("/workspace/datasets/yolo_dataset/images" + "/" + filename)
        txt_name = filename.split('.')[0] + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(labels_dir, txt_name), 'w')

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            ignore = ann["ignore"]
            bbox = cocobbox2yolobbox(ann["bbox"])
            f_txt.write("%s %s %s %s %s %s\n" % (ann['category_id'], bbox[0], bbox[1], bbox[2], bbox[3], ignore))
        f_txt.close()

    # 将数据集写入文件
    with open(text_data_fp, 'w') as f:
        for line in text_data:
            f.write(line + '\n')
    print('labels create done.')

    for i in trange(m):
        img_id = img_ids[i]
        filename = imgs[img_id]['file_name']
        shutil.copy(os.path.join(origin_imgs_dir, filename), os.path.join(imgs_dir, filename))
    print('images copy done.')


def coco2yolo_type(dataset_type):
    save_dir = os.path.join(root_dir, 'yolo_dataset')
    if dataset_type == 'train':
        json_fp = os.path.join(root_dir, 'coco', 'annotations', 'train.json')
        imgs_dir = os.path.join(root_dir, 'coco', 'train')
        coco2yolo(dataset_type, json_fp, imgs_dir, save_dir)
    elif dataset_type == 'val':
        json_fp = os.path.join(root_dir, 'coco', 'annotations', 'val.json')
        imgs_dir = os.path.join(root_dir, 'coco', 'val')
        coco2yolo(dataset_type, json_fp, imgs_dir, save_dir)
    elif dataset_type == 'test':
        json_fp = os.path.join(root_dir, 'coco', 'annotations', 'test.json')
        imgs_dir = os.path.join(root_dir, 'coco', 'test')
        coco2yolo(dataset_type, json_fp, imgs_dir, save_dir)


if __name__ == '__main__':
    coco2yolo_type('train')
    coco2yolo_type('val')
    # coco2yolo_type('test')
