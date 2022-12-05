"""
   （1）将['other', ' other']转换为iscrowd
   （2）根据map更改标注中的类
   :param anno_list: 所有标注文件的list
   :param class_map: 类型映射关系
   :return:
   type: ['两轮车（骑行）', '特殊目标（固定的）', '轿车', '货车', '行人', '三轮车', '大特种车', '工程车', '巴士',
         '无人骑行（停靠）', '小特种车', '京东小车', '障碍物（可活动的）', '其他可移动物', '其他特种车', '动物']
   """

import os
from lxml import etree
from loguru import logger
import json
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

"""
最终需要的类
"""
CLASSES = [
        "Big car",     # 大机动车
        "Small car",  # 小机动车
        "Non motor",     # 非机动车
        "Pedestrian",    #行人
        "Obstacle",  # 障碍物
    ]

def class_filter(anno_list, class_map=None):
    """
    （1）将['other', ' other']转换为iscrowd
    （2）根据map更改标注中的类
    :param anno_list: 所有标注文件的list
    :param class_map: 类型映射关系
    :return:
    """
    #类别影射
    '''
    type: ['两轮车（骑行）', '特殊目标（固定的）', '轿车', '货车', '行人', '三轮车', '大特种车', '工程车', '巴士',
         '无人骑行（停靠）', '小特种车', '京东小车', '障碍物（可活动的）', '其他可移动物', '其他特种车', '动物']
    '''

    sub_type_map = {
        'ride': [
            '两轮车（骑行）'
        ],
        'special target': [
            '特殊目标（固定的）'
        ],
        'car': [
            '轿车',
        ],
        'truck': [
            '货车'
        ],
        'pedestrian': [
            '行人',
        ],
        'Tricycle': [
            '三轮车'
        ],
        'big special vehicle': [
            '大特种车'
        ],
        'engineering vehicler': [
            '工程车',
        ],
        'bus': [
            '巴士'
        ],
        'dock': [
            '无人骑行（停靠）',
        ],
        'Small special vehicle': [
            '小特种车'
        ],
        'Jingdong car': [
            '京东小车',
        ],
        'movable': [
            '障碍物（可活动的）'
        ],
        'other movable objects': [
            '其他可移动物'
        ],
        'Other special vehicles': [
          '其他特种车'
        ],
        'anlmal': [
            "动物"
        ]

    }

    pbar = tqdm(anno_list, desc="filter mapping")
    for echo_anno in pbar:
        for idx in range(len(echo_anno["shapes"])-1, -1, -1):     # 因为在遍历过程中会删除元素，所以逆序遍历
            obj = echo_anno["shapes"][idx]
            type = obj["type"]

            # face = obj['face'] if obj['face'] != '' else "None"

            isMatch = False # 记录标签是否匹配到了map中的值
            for k, v in sub_type_map.items():
                if type in v:
                    type = obj["type"] = k
                    isMatch = True
                    break


            assert isMatch, f"{type} 没有匹配的map值"
        pass

    pass


def save2xml(anno_list, voc_anno_dir, src_img_dir, dst_img_dir, args, proess_type="train") -> list:
    """
    """
    name_list = []
    folder_info = args.folder_info
    source_database_info = args.source_database_info
    source_annotation_info = args.source_annotation_info

    pbar = tqdm(anno_list, desc=f"{proess_type}...")
    for each_anno in pbar:
        image_name = each_anno['image']["image_name"]

        try:
            metadata_weather = each_anno['image']["metadata"]['weather']
            metadata_lightning = each_anno['image']["metadata"]['lightning']
        except:
            pass
        base_name = ".".join(image_name.split(".")[:-1])  # 去掉最后的 .扩展名
        image_type = image_name.split(".")[-1]  # 获取图片类型用于格式转化，暂时未用；现在用直接改名的方式
        image_target_name = base_name + ".jpg"  # 都需要改为.jpg格式
        name_list.append(base_name)
        # 保存xml
        xml_name = base_name + ".xml"
        dst_xml_path = os.path.join(voc_anno_dir, xml_name)
        root = etree.Element("annotation")
        etree.SubElement(root, "folder").text = folder_info
        etree.SubElement(root, "filename").text = image_target_name

        source = etree.SubElement(root, "source")
        etree.SubElement(source, "database").text = source_database_info
        etree.SubElement(source, "annotation").text = source_annotation_info
        etree.SubElement(source, "metadata_weather").text = metadata_weather
        etree.SubElement(source, "metadata_lightning").text = metadata_lightning
        size = etree.SubElement(root, "size")
        # 数据集中很多标注的宽高都是0
        image_width = 1920
        image_height = 1080
        assert image_width > 0 and image_height > 0, f"{base_name}标注文文件中图片宽高信息异常"
        # if image_width>0 and image_height>0: print(f"{base_name}标注文文件中图片宽高信息异常")
        etree.SubElement(size, "width").text = str(image_width)
        etree.SubElement(size, "height").text = str(image_height)
        etree.SubElement(size, "depth").text = str(3)

        for obj_anno in each_anno["shapes"]:
            obj = etree.SubElement(root, "object")
            if obj_anno["type"] is None:
                etree.SubElement(obj, "name").text = "NULL"
            else:
                etree.SubElement(obj, "name").text = obj_anno["type"]
            bndbox = etree.SubElement(obj, "bndbox")
            cx = float(obj_anno["shape_attributes"]["x"])    # cx
            cy = float(obj_anno["shape_attributes"]["y"])    # cy
            w  = float(obj_anno["shape_attributes"]["width"])
            h  = float(obj_anno["shape_attributes"]["height"])
            etree.SubElement(bndbox, "xmin").text = str(round(cx))
            etree.SubElement(bndbox, "ymin").text = str(round(cy))
            etree.SubElement(bndbox, "xmax").text = str(round(cx + w))
            etree.SubElement(bndbox, "ymax").text = str(round(cy + h))

        tree = etree.ElementTree(root)
        # with open(dst_xml_path, "w", encoding="utf-8") as wf:
        #     tree.write(wf, pretty_print=True)
        tree.write(dst_xml_path, pretty_print=True)

        # # 复制图片
        src_img_path = os.path.join(src_img_dir, image_name)
        # dst_img_path = os.path.join(img_dir, image_name)
        dst_img_path = os.path.join(dst_img_dir, image_target_name)
        shutil.copy(src_img_path, dst_img_path)

    return name_list


def make_argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/home/chenzhen/code/detection/datasets/dt_imgdata", help="数据集的根目录")
    parser.add_argument('--folder_info', default="VOC_DT_20221111", help="xml中文件夹名，并以此为名为目录保存数据")
    parser.add_argument('--source_database_info', default="img_data", help="数据来源")
    parser.add_argument('--source_annotation_info', default="DT", help="数据标注者")
    parser.add_argument('--voc_save_path', default="/home/chenzhen/code/detection/datasets/dt_imgdata", help="voc格式数据保存目录")
    parser.add_argument('--data_split_path', default=None, help="数据分割指定的目录")

    return parser

@logger.catch()
def main():
    parser = make_argParse()
    args = parser.parse_args()

    dataset_path = args.dataset_path
    imgs_path = os.path.join(dataset_path, "imgs")
    labels_path = os.path.join(dataset_path, "labels")

    # 获取标注信息
    anno_list = []
    for file in os.listdir(labels_path):
        file_path = os.path.join(labels_path, file)
        with open(file_path, "r", encoding="utf-8") as rf:
            echo_anno = json.load(rf)
            anno_list.append(echo_anno)

    # 标签重分配
    class_filter(anno_list)

    # 构建voc目录
    voc_save_dir = args.voc_save_path
    voc_folder = args.folder_info
    voc_anno_dir = os.path.join(voc_save_dir, voc_folder, "Annotations")
    voc_img_dir = os.path.join(voc_save_dir, voc_folder, "JPEGImages")
    train_val_dir = os.path.join(voc_save_dir, voc_folder, "ImageSets", "Main")
    if not os.path.exists(voc_anno_dir):
        os.makedirs(voc_anno_dir)
    if not os.path.exists(voc_img_dir):
        os.makedirs(voc_img_dir)
    if not os.path.exists(train_val_dir):
        os.makedirs(train_val_dir)

    # 将数据写入xml
    all_name_list = save2xml(anno_list, voc_anno_dir, imgs_path, voc_img_dir, args, "all")

    # 数据集分割
    if args.data_split_path is None:
        train_name_list, test_name_list = train_test_split(all_name_list, train_size=0.8, test_size=0.2, random_state=0, shuffle=True)
    train_name_path = os.path.join(train_val_dir, "train.txt")
    test_name_path = os.path.join(train_val_dir, "test.txt")
    with open(train_name_path, "w", encoding="utf-8") as wf:
        for name in tqdm(train_name_list, desc="save train_name_list"):
            wf.write(name+"\n")
    with open(test_name_path, "w", encoding="utf-8") as wf:
        for name in tqdm(test_name_list, desc="save test_name_list"):
            wf.write(name+"\n")

if __name__ == "__main__":
    main()
    pass