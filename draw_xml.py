import xml.etree.ElementTree as ET  # 读取xml。
import os
from PIL import Image, ImageDraw, ImageFont
import uuid

def parse_rec(filename):
    tree = ET.parse(filename)  # 解析读取xml函数
    objects = []
    img_dir = []
    for xml_name in tree.findall('filename'):
        img_path = os.path.join(pic_path, xml_name.text)
        img_dir.append(img_path)
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['visibility'] = obj.find('visibility').text
        obj_struct['occ'] = int(obj.find('occ').text)
        obj_struct['direct'] = int(obj.find('direct').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects, img_dir

# 可视化

def visualise_gt(objects, img_dir):
    save_path = "/home/chenzhen/code/detection/datasets/dt_imgdata/Pedestrian-imgs/"
    for id, img_path in enumerate(img_dir):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        for a in objects:
            xmin = int(a['bbox'][0])
            ymin = int(a['bbox'][1])
            xmax = int(a['bbox'][2])
            ymax = int(a['bbox'][3])
            if a['name'] == "Pedestrian":
            # label = a['name']
            # visibility = a['visibility']
            # occ = a['occ']
            # direct = a['direct']
                draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline='red', width=2)
            # draw.text((xmin - 10, ymin - 15), label, fill='#0504aa')  # 利用ImageDraw的内置函数，在图片上写入文字
            # draw.text((xmin - 20, ymin - 15), str(visibility), fill='#C504AA')
            # draw.text((xmin - 30, ymin - 15), str(occ), fill='#C5F4AA')
            # draw.text((xmin - 40, ymin - 15), str(direct), fill='#5504AA')

        # img.show()
            finename = save_path + str(img_path.split("/")[-1:][0])
            img.save(finename)


ann_path = "/home/chenzhen/code/detection/datasets/dt_imgdata/val_xml_label" # xml文件所在路径
pic_path = "/home/chenzhen/code/detection/datasets/dt_imgdata/coco_dt/val"  # 样本图片路径


for filename in os.listdir(ann_path):
    xml_path = os.path.join(ann_path, filename)
    object, img_dir = parse_rec(xml_path)
    visualise_gt(object, img_dir)