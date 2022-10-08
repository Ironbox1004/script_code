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
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects, img_dir

# 可视化

def visualise_gt(objects, img_dir):
    save_path = r"D:\dt_detection\datasets\results\\"
    for id, img_path in enumerate(img_dir):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        for a in objects:
            xmin = int(a['bbox'][0])
            ymin = int(a['bbox'][1])
            xmax = int(a['bbox'][2])
            ymax = int(a['bbox'][3])
            label = a['name']
            draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline='#0504aa', width=2)
            draw.text((xmin - 10, ymin - 15), label, fill='#0504aa', font=font)  # 利用ImageDraw的内置函数，在图片上写入文字
        # img.show()
        finename = save_path + str(uuid.uuid1()) + ".jpg"
        img.save(finename)



fontPath = "C:\Windows\Fonts\Consolas\consola.ttf"  # 字体路径
ann_path = r"D:\dt_detection\datasets\test_xml" # xml文件所在路径
pic_path = r"D:\dt_detection\datasets\test_img"  # 样本图片路径
font = ImageFont.truetype(fontPath, 16)

for filename in os.listdir(ann_path):
    xml_path = os.path.join(ann_path, filename)
    object, img_dir = parse_rec(xml_path)
    visualise_gt(object, img_dir)