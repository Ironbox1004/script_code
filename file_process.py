
import shutil
import os

import json
import os
import shutil


img_dir = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/dair-hz-1/train"

val_ann_dir = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/dair-hz-1/random_coco_split_val.json"
train_ann_dir = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/dair-hz-1/random_coco_split_train.json"

val_save_dir = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/dair-hz-1/val_use"
train_save_dir = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/dair-hz-1/train_use"


with open(train_ann_dir, 'r', encoding='unicode_escape') as fp:
    img_list = os.listdir(img_dir)
    json_data = json.load(fp)
    for i in range(len(json_data['images'])):
        if json_data['images'][i]['file_name'] in img_list:
            shutil.copy(os.path.join(img_dir, json_data['images'][i]['file_name']),
                        os.path.join(train_save_dir, json_data['images'][i]['file_name']))
        else:
            print(json_data['images'][i]['file_name'])






# img_path = '/home/chenzhen/code/detection/datasets/dt_imgdata/640-640-result-1'
# xml_path = '/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221115/Annotations/'
# save_xml_path = '/home/chenzhen/code/detection/datasets/dt_imgdata/val_xml_label'
#
# def copy_file(img_path,xml_path,save_xml_path):
#     for img_list in os.listdir(img_path):
#         img_name = img_list.split('.')
#         if len(img_name) == 2:
#             copy_name = xml_path + str(img_name[0])  + '.xml'
#             shutil.copy(copy_name, save_xml_path)
#         else:
#             copy_name = xml_path + str(img_name[0]) + '.' + str(img_name[1]) + '.xml'
#         shutil.copy(copy_name, save_xml_path)

#
# def file_rename():
#     path = "/home/chenzhen/code/detection/datasets/Dair_x2x/single-infrastructure-side/calib/camera_intrinsic"
#
#     # 获取该目录下所有文件，存入列表中
#     fileList = os.listdir(path)
#     from os.path import join as pjoin
#     n = 0
#     for i in fileList:
#         # 设置旧文件名（就是路径+文件名）
#         oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符
#
#         # 设置新文件名
#         newname = path + os.sep + 'single_' + i
#         # print(newname)
#
#         os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
#         print(oldname, '======>', newname)
#
#         n += 1
#
# def save_file():
#     file_path = r"D:\detection\datasets\test_img"
#     copy_file_path = r"D:\detection\datasets\union2voc_multiClass\VOCdevkit\VOC_UnDt20220823\process\\"
#     to_path = r"D:\detection\datasets\test_xml"
#
#     for filename in os.listdir(file_path):
#         name_list = filename.split(".")[0:-1]
#         if len(name_list) == 2:
#             from_path_name = copy_file_path + str(name_list[0]) + "." + str(name_list[1]) + ".xml"
#             # print(from_path_name)
#             # copy(from_path_name,to_path)
#         else:
#             from_path_name = copy_file_path + str(name_list[0]) + ".xml"
#             # print(from_path_name)
#             copy(from_path_name, to_path)
#
# def txt_chongfu():
#     train_text = "/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221114/ImageSets/Main/train.txt"
#     val_text = "/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221114/ImageSets/Main/test.txt"
#
#     train_list = []
#     val_list = []
#
#     with open(train_text, 'r') as f:
#         for line in f:
#             list = line[:-1]
#             file_name = (list.split('\\')[-1]).split('.')
#             # if len(file_name) == 2:
#             #     copy_name = file_name[0]
#             # else:
#             copy_name = file_name[0] + '.' + file_name[1]
#
#             train_list.append(copy_name)
#
#     with open(val_text, 'r') as f:
#         for line in f:
#             list = line[:-1]
#             file_name = (list.split('\\')[-1]).split('.')
#             # if len(file_name) == 2:
#             #     copy_name = file_name[0]
#             # else:
#             copy_name = file_name[0] + '.' + file_name[1]
#
#             val_list.append(copy_name)
#
#     num = 0
#     for i in val_list:
#         if i in train_list:
#             print(i)
#             num = num + 1
#             print(num)

# if __name__ == '__main__':
#
#     path = "/home/chenzhen/code/detection/datasets/plate_plt/val_detect/double_val/"
#     train_images = "/home/chenzhen/code/detection/datasets/yolo_plate/images/val"
#     train_label = "/home/chenzhen/code/detection/datasets/yolo_plate/labels/val"
#
#     for img_list in os.listdir(path):
#         text = os.path.splitext(img_list)[-1]
#         copy_name = path + img_list
#         if text == ".jpg":
#             shutil.copy(copy_name, train_images)
#         else:
#             shutil.copy(copy_name, train_label)


    # img_path = '/home/chenzhen/code/detection/datasets/repo3d/val/extrinsics'
    # xml_path = '/home/chenzhen/code/detection/datasets/repo3d/val/validation-image_2/'
    # save_xml_path = '/home/chenzhen/code/detection/datasets/repo3d/val/images'
    #
    # for img_list in os.listdir(img_path):
    #     img_name = img_list.split('.')
    #     # if len(img_name) == 2:
    #     #     copy_name = xml_path + str(img_name[0])  + '.xml'
    #     #     shutil.copy(copy_name, save_xml_path)
    #     # else:
    #     copy_name = xml_path + str(img_name[0]) + '.jpg'
    #     try:
    #         shutil.copy(copy_name, save_xml_path)
    #     except:
    #         print(copy_name)

    # train_text = "/home/chenzhen/code/detection/datasets/repo3d/val/extrinsics"
    # val_text = "/home/chenzhen/code/detection/datasets/repo3d/val/validation-image_2"
    #
    # train_list = []
    # val_list = []
    #
    # for img_list in os.listdir(train_text):
    #     img_name = img_list.split('.')
    #     train_list.append(img_name[0])
    #
    # for img_list1 in os.listdir(train_text):
    #     img_name = img_list1.split('.')
    #     val_list.append(img_name[0])
    #
    # num = 0
    # for i in train_list:
    #    if i not in val_list:
    #         print(i)
    #         num = num + 1
    #         print(num)

#
# if __name__ == '__main__':
#     train = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/dair_hz/train"
#     val = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/dair_hz/val"
#
#     train_list = os.listdir(train)
#     val_list = os.listdir(val)
#
#     for i in train_list:
#         if i in val_list:
#             print(i)