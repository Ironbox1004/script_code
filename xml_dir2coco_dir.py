

import json
import os
import shutil
import datetime
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import trange

root_dir = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/copy_data"
img_path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/copy_data/pic"
xml_path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/copy_data/xml"

class_name_to_id = {'Car': 0, 'Bus': 1, 'Cyclist': 2, 'Pedestrian': 3,
                    'driverless_car': 4, 'Truck': 5, 'Tricyclist': 6, 'Trafficcone': 7}
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
        dict(supercategory=None, id=id, name=name,)
    )

images = os.listdir(img_path)
images_id = {}
for idx, image_name in enumerate(images):
    images_id.update({image_name[:-4]: idx})

train_img = []
for file_name in os.listdir(img_path):
    train_img.append(file_name)
for img in train_img:
    data["images"].append(
        dict(
            license=0,
            url=None,
            file_name=img,  # 图片的文件名带后缀
            height=1920,
            width=1080,
            date_captured=None,
            # id=image[:-4],
            id=images_id[img[:-4]],
        )
    )

train_xml = [i[:-4] + '.xml' for i in train_img]
bbox_id = 0
for xml in train_xml:
    category = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    ignore_list = []
    tree = ET.parse(os.path.join(xml_path, xml))
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

json.dump(data, open(os.path.join(root_dir,  'train1.json'), 'w'))