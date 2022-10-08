
train_text = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/yolo_dataset/ImageSets/train.txt"
val_text = "/home/chenzhen/code/detection/datasets/union2voc_multiClass/yolo_dataset/ImageSets/val.txt"

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