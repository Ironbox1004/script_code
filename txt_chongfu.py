
train_text = "/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221026/ImageSets/Main/train.txt"
val_text = "/home/chenzhen/code/detection/datasets/dt_imgdata/VOC_DT_20221026/ImageSets/Main/test.txt"

train_list = []
val_list = []

with open (train_text, 'r') as f:
    for line in f :
        list = line[:-1]
        file_name = (list.split('\\')[-1]).split('.')
        if len(file_name) == 2:
            copy_name = file_name[0]
        else:
            copy_name = file_name[0] + '.' + file_name[1]

        train_list.append(copy_name)

with open (val_text, 'r') as f:
    for line in f :
        list = line[:-1]
        file_name = (list.split('\\')[-1]).split('.')
        if len(file_name) == 2:
            copy_name = file_name[0]
        else:
            copy_name = file_name[0] + '.' + file_name[1]

        val_list.append(copy_name)

for i in val_list:
    if i in train_list:
        print(i)