import json
from tqdm import tqdm
import os
# 数据路径
anno_list = []
labels_path = "/home/chenzhen/code/detection/datasets/dt_imgdata/labels"
# 读取文件数据

json_type = []
json_original_type = []

for file in os.listdir(labels_path):
    file_path = os.path.join(labels_path, file)
    with open(file_path, "r", encoding="utf-8") as rf:
        echo_anno = json.load(rf)
        anno_list.append(echo_anno)
    # print(anno_list)

print(len(anno_list))
pbar = tqdm(anno_list, desc="filter mapping")
for echo_anno in pbar:
    for idx in range(len(echo_anno["shapes"])-1, -1, -1):     # 因为在遍历过程中会删除元素，所以逆序遍历
        obj = echo_anno["shapes"][idx]
        type = obj["type"]
        original_type = obj["original_type"]
        if type not in json_type:json_type.append(type)
        if original_type not in json_original_type:json_original_type.append(original_type)
print("type:", json_type)
print("json_original_type", json_original_type)
