import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import os
import time
import json


def ConvertVOCXml(file_path="", file_name=""):
    xml_file = open(('/home/chenzhen/code/detection/datasets/Dair_x2x/single-infrastructure-side-xml/' + file_name[:-5] + '.xml'), 'w')

    f2 = open(('/home/chenzhen/code/detection/datasets/Dair_x2x/single-infrastructure-side/calib/camera_intrinsic/' + file_name))
    content2 = json.load(f2)

    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>cooperative-vehicle-infrastructure</folder>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + '1920' + '</width>\n')
    xml_file.write('        <height>' + '1080' + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('        <date_captured>' + str(content2['cameraID']) + '</date_captured>\n')
    xml_file.write('    </size>\n')

    f = open(('/home/chenzhen/code/detection/datasets/Dair_x2x/single-infrastructure-side/label/camera/' + file_name))
    content = json.load(f)

    for i in range(len(content)):
        gt_label = content[i]['type']
        gt_direct = content[i]['truncated_state']
        gt_occ = content[i]['occluded_state']
        xmin = content[i]['2d_box']['xmin']
        ymin = content[i]['2d_box']['ymin']
        xmax = content[i]['2d_box']['xmax']
        ymax = content[i]['2d_box']['ymax']
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)

        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(gt_label) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <direct>' + str(gt_direct) + '</direct>\n')
        xml_file.write('        <occ>' + str(gt_occ) + '</occ>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(xmin) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(ymin) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(xmax) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(ymax) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')
    xml_file.close()


if __name__ == "__main__":
    basePath = '/home/chenzhen/code/detection/datasets/Dair_x2x/single-infrastructure-side/label/camera/'  # dair-v2x数据集的json文件放置位置
    totaljson = os.listdir(basePath)
    totaljson.sort()
    total_num = 0
    flag = False
    print("正在转换")
    saveBasePath = '/home/chenzhen/code/detection/datasets/Dair_x2x/single-infrastructure-side-xml//'  # voc格式的xml文件放置位置
    if os.path.exists(saveBasePath) == False:  # 判断文件夹是否存在
        os.makedirs(saveBasePath)

    # ConvertVOCXml(file_path="samplexml",file_name="000009.xml")
    # Start time
    start = time.time()
    for json1 in totaljson:
        file_name = os.path.join(basePath, json1)
        print(file_name)

        ConvertVOCXml(file_path=saveBasePath, file_name=json1)

    # End time
    end = time.time()
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))
