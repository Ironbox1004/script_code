
import os
import xml.etree.ElementTree as ET
from os.path import join as pjoin
from tqdm import tqdm

src = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ignore_xml_1"
# area_poly = [[1263,85],[1482,275]]
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

for category_name in tqdm(os.listdir(src)):
    annotations_path = pjoin(src, category_name)
    # print(annotations_path)
    tree = ET.parse(annotations_path)
    root = tree.getroot()
    objs = root.findall('object')
    xyxy = []
    for obj in objs:
        if obj.find("name").text == "Barrowlist":
            # bbox = obj.find("bndbox")
            # xyxy = [int(bbox.find("xmin").text), int(bbox.find("ymin").text), int(bbox.find("xmax").text), int(bbox.find("ymax").text)]
            # # print(xyxy)
            # area = (int(bbox.find("xmax").text) - int(bbox.find("xmin").text)) * (int(bbox.find("ymax").text) - int(bbox.find("ymin").text))
            # if in_poly_area_dangerous(xyxy, area_poly) == True:
            #     if area > 5000:
            #         print(area)
            root.remove(obj)
        else:
            pass
    tree.write(annotations_path,encoding='utf-8')