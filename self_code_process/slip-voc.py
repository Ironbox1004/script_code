import json
import os
import random
import time
import shutil
import glob

root_path = "/home/chenzhen/code/detection/datasets/Dair_x2x"
train_percent = 0.8

def voc_dataset_split():
    file_train = open(
        os.path.join(root_path,  "cooperative-vehicle-infrastructure", "ImageSets", "Main", "train.txt"), 'w')
    file_val = open(
        os.path.join(root_path,  "cooperative-vehicle-infrastructure", "ImageSets", "Main", "val.txt"), 'w')

    xml_total_filename = glob.glob(os.path.join(root_path, "cooperative-vehicle-infrastructure", "*.xml"))
    for idx, xml in enumerate(xml_total_filename):
        xml_total_filename[idx] = xml.split('\\')[-1]
    num_total = len(xml_total_filename)
    num_train = int(num_total*train_percent)
    train_sample = random.sample(xml_total_filename, num_train)

    for name in xml_total_filename:
        if name in train_sample:
            file_train.write(name[:-4]+'\n')
        else:
            file_val.write(name[:-4]+'\n')

    file_train.close()
    file_val.close()

if __name__ == '__main__':
    voc_dataset_split()