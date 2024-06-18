"""
IMPORTANT!!

Data is from Met-Meme dataset: https://github.com/liaolianfoka/MET-Meme-A-Multi-modal-Meme-Dataset-Rich-in-Metaphors
Please refer to this link for specific data formats.
Since the original data did not clearly specify the method of dividing the dataset, the purpose of this file is to tell you how we did a simple data segmentation.
The splitting ratio is 6:2:2 for the training set, validation set, and testing set.

You can try your own way to divide the dataset. It is sufficient as long as the final data format matches the data format in the data/.
"""


import csv
import json
import random

random.seed(42)

load_path = '''
data path you donwload from met-meme dataset, for example:
data/Chinese -> This contains tow file.
    |_C_text.csv -> This file indicates the OCR text corresponding to each image
    |_label_C.csv -> This file indicates the label to each image
'''


def read_file(file_name, image_file, chinese=False):
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        output_list = []
        for i, line in enumerate(reader):
            if i == 0:
                head = line
            else:
                if chinese:
                    line[0] = image_file + line[0].replace('_', '- ')
                else:
                    line[0] = image_file + line[0]
                output_list.append(line)
            random.shuffle(output_list)
        return output_list, head


def device_dataset(data):
    train_num = int(len(data) * 0.6)
    train_data = data[: train_num]
    val_num = int(len(data) * 0.2)
    val_data = data[train_num: train_num + val_num]
    test_data = data[val_num + train_num:]
    return train_data, val_data, test_data


def write_file(file_name, data, head):
    with open(file_name, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for line in data:
            writer.writerow(line)


def add_text(data, text_data, head):
    for line in data:
        img_name = line[0]
        sentence = text_data[img_name]
        line.insert(1, sentence)
    head.insert(1, 'text')
    return data, head


def convert_list2dict(data, head):
    new_data = []
    for line in data:
        new_data.append({head[i]: line[i] for i in range(len(line))})
    return new_data


def save_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data))


chinese_text_data, _ = read_file(f'{load_path}/C_text.csv', 'Chinese/', chinese=True)
chinese_text_data = {line[0]: line[1] for line in chinese_text_data}
english_text_data, _ = read_file(f'{load_path}/E_text.csv', 'English/')
english_text_data = {line[0]: line[1] for line in english_text_data}

chinese_data, c_head = read_file(f'{load_path}/label_C.csv', 'Chinese/', chinese=True)
chinese_data, c_head = add_text(chinese_data, chinese_text_data, c_head)
english_data, e_head = read_file(f'{load_path}/label_E.csv', 'English/')
english_data, e_head = add_text(english_data, english_text_data, e_head)


c_train, c_val, c_test = device_dataset(chinese_data)
e_train, e_val, e_test = device_dataset(english_data)

train_data = c_train + e_train
random.shuffle(train_data)
val_data = c_val + e_val
random.shuffle(val_data)
test_data = c_test + e_test
random.shuffle(test_data)

save_json('../data/train_data.json', convert_list2dict(train_data, c_head))
save_json('../data/val_data.json', convert_list2dict(val_data, c_head))
save_json('../data/test_data.json', convert_list2dict(test_data, c_head))
