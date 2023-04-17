import json
import os

import cv2


# CLASSES = [
#         "Car",
#         "Bus",
#         "Cycling",
#         "Pedestrian",
#         "driverless_Car",
#         "Truck",
#         "Animal",
#         "Obstacle",
#         "Special_Target",
#         "Other_Objects",
#         "Unmanned_riding"
#     ]
CLASSES = [
            "Car",
            "Bus",
            "Cycling",
            "Pedestrian",
            "Special_Car",
            "Truck",
            "Obstacle",
            "Special_Target",
            "Other_Objects"
]
class CocoDataVisualization:
    def __init__(self, imgPath, jsonPath):
        self.imgPath = imgPath
        self.jsonPath = jsonPath

    def visualize(self, out, color=(0, 255, 255), thickness=1):
        with open(self.jsonPath, 'r') as f:
            annotation_json = json.load(f)

        for img in annotation_json['images']:
            image_name = img['file_name']  # 读取图片名
            id = img['id']  # 读取图片id
            image_path = os.path.join(self.imgPath, str(image_name).zfill(5))  # 拼接图像路径
            image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像
            num_bbox = 0  # 统计一幅图片中bbox的数量

            for i in range(len(annotation_json['annotations'][::])):
                if annotation_json['annotations'][i - 1]['image_id'] == id:
                    num_bbox = num_bbox + 1
                    x, y, w, h = annotation_json['annotations'][i - 1]['bbox']  # 读取边框
                    label = CLASSES[annotation_json['annotations'][i - 1]['category_id']]
                    image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=color,
                                          thickness=thickness)
                    image = cv2.putText(image, str(label), (int(x + w), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            cv2.imwrite(os.path.join(out, image_name), image)


if __name__ == '__main__':
    # the first param is the directory's path of images
    # the second param is the path of json file
    d = CocoDataVisualization('/home/chenzhen/code/detection/datasets/dair_hz/train',
                              '/home/chenzhen/code/detection/datasets/dair_hz/annotations/train.json')

    # this param is the output path
    d.visualize('/home/chenzhen/code/detection/datasets/dair_hz/out')


