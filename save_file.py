
import os
from shutil import copy

file_path = r"D:\detection\datasets\test_img"
copy_file_path = r"D:\detection\datasets\union2voc_multiClass\VOCdevkit\VOC_UnDt20220823\process\\"
to_path = r"D:\detection\datasets\test_xml"

for filename in os.listdir(file_path):
    name_list = filename.split(".")[0:-1]
    if len(name_list) == 2:
        from_path_name = copy_file_path + str(name_list[0]) + "." + str(name_list[1]) + ".xml"
        # print(from_path_name)
        # copy(from_path_name,to_path)
    else:
        from_path_name = copy_file_path + str(name_list[0]) + ".xml"
        # print(from_path_name)
        copy(from_path_name, to_path)