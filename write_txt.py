

text_data = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ImageSets/Main/val.txt"
path = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ImageSets/Main/"
data_list = []
save_file = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/VOCdevkit/VOC_UnDt20220823/ImageSets/Main/new.txt"

with open (text_data, 'r') as f:
    for line in f :
        list = line[:-1]
        file_name = (list.split('/')[-1]).split('.')

        if len(file_name) == 2:
            copy_name = str(file_name[0])
        else:
            copy_name =  str(file_name[0]) + '.' + str(file_name[1])
        # copy_name = xml_path + '\\' + file_name +'.xml'
        data_list.append(copy_name)

with open(save_file, 'w') as f:
    for line in data_list:
        print(line)
        f.write(line + '\n')