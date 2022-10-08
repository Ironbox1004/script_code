#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-08-22 17:21
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : file_rename.py

import os

path =r"E:\download\DAIR-V2X-I\single-infrastructure-side-image_8643452398735360\single-infrastructure-side-image"

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
from os.path import join as pjoin
n = 0
for i in fileList:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符

    # 设置新文件名
    newname = path + os.sep +'V2X_I_'+ i
    # print(newname)

    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

    n += 1