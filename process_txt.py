#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-09-01 11:20
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : process_txt.py

import os
  # 整行读取

f = open(r'E:\detection\datasets\union2voc_multiClass\yolo_dataset\\test.txt')
lines = f.readlines()          # 整行读取
f.close()
for line in lines:
    rs = line.rstrip('\n')     # 去除原来每行后面的换行符，但有可能是\r或\r\n
    print(rs)
    # newname = rs.replace(rs, './data/coco'+rs)
    # newfile = open('./trainvalno5k1.txt', 'a')
    # newfile.write(newname+'\n')
    # newfile.close

