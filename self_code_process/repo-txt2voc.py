import os
import yaml
import time
from tqdm import tqdm

def ConvertVOCXml(file_name=None, xml_file=None, ymal_path=None, label_file=None):
    xml_file = open((os.path.join(xml_file, file_name) + ".xml"), 'w')

    yamlPath = os.path.join(ymal_path, file_name) + ".yaml"
    f = open(yamlPath, 'r')
    cont = f.read()
    x = yaml.load(cont, Loader=yaml.FullLoader)

    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>repo-3d</folder>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + '1920' + '</width>\n')
    xml_file.write('        <height>' + '1080' + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('        <date_captured>' + str(x['child_frame_id']) + '</date_captured>\n')
    xml_file.write('    </size>\n')

    gt_path = os.path.join(label_file, file_name) + ".txt"
    file = open(gt_path, 'r')
    label = file.readlines()
    for gt in label:
        gt = gt.split()
        category_name = gt[0].capitalize()
        if category_name == 'Unknown_unmovable':
            category_name = 'Trafficcone'
        elif category_name == 'Barrow':
            category_name = 'Barrowlist'
        elif category_name == 'Unknowns_movable':
            continue
        truncate = int(gt[1])
        occ = int(gt[2])
        iscrowd = 0
        direct = 0
        xyxy = [
            float(gt[4]),
            float(gt[5]),
            float(gt[6]) - float(gt[4]),
            float(gt[7]) - float(gt[5])
            ]
        # float(gt[4]),
        # float(gt[5]),
        # float(gt[6]),
        # float(gt[7])

        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(category_name) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncate>' + str(truncate) + '</truncate>\n')
        xml_file.write('        <occ>' + str(occ) + '</occ>\n')
        xml_file.write('        <direct>' + str(direct) + '</direct>\n')
        xml_file.write('        <iscrowd>' + str(iscrowd) + '</iscrowd>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(xyxy[0]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(xyxy[1]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(xyxy[2]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(xyxy[3]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')
    xml_file.close()



if __name__ == '__main__':
    xml_file = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/repo3d/train/xml_label"
    ymal_path = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/repo3d/train/copy-extrinsic"
    label_file = "/home/chenzhen/code/detection/datasets/hz_baidu_dataset/repo3d/train/copy-label_2"
    # ConvertVOCXml(file_path="samplexml",file_name="000009.xml")
    # Start time
    start = time.time()
    for file_name in tqdm(os.listdir(label_file)):
        file_name = file_name.split('.')[0]
        ConvertVOCXml(file_name=file_name, xml_file=xml_file, ymal_path=ymal_path, label_file=label_file)
    # End time
    end = time.time()
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))


