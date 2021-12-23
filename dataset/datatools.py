#!/usr/bin/env python3


import pathlib
import json
from random import sample
import sys

# patterns = ['* T*.gz', '*-T*.gz', '*T1*.gz']


def get_files(pathstr, img_pattern, label_patterns):
    """Get image-label pair from sepcific label_patterns and a single image_pattern.

    Args:
        pathstr (str): dataset path. e.g, '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211214/shidaoai/sichuan'
        img_pattern (str): e.g, '*CT*.gz'
        label_patterns (Union[str]): label patterns. e.g, label_patterns=['* T*.gz', '*-T*.gz', '*T1*.gz']

    Returns:
        [type]: [description]
    """
    path = pathlib.Path(pathstr)
    images = []
    labels = []
    for pattern in label_patterns:
        for item in list(path.rglob(pattern)):
            if not item.exists():
                continue
            label = item.as_posix()
            image = list(item.parent.rglob(img_pattern))
            if not image:
                continue
            image = image[0].as_posix()
            images.append(image)
            labels.append(label)
    return images, labels


def move_img_label(imgs, labels, img_folder, label_folder):
    """[summary]

    Args:
        imgs (list(str)): image path strings
        labels (list(str)): label path strings
        img_folder (str): destination folder for images
        label_folder (str): destination folder for labels
    """
    img_folder_obj = pathlib.Path(img_folder)
    if not img_folder_obj.exists():
        img_folder_obj.mkdir(parents=True)

    label_folder_obj = pathlib.Path(label_folder)
    if not label_folder_obj.exists():
        label_folder_obj.mkdir(parents=True)

    for img, label in zip(sorted(imgs), sorted(labels)):

        img_obj = pathlib.Path(img)
        label_obj = pathlib.Path(label)

        img_target = img_folder + '/' + img_obj.name
        img_obj.rename(img_target)
        # print(img_obj.as_posix(), img_obj.rename(img_target))

        label_target = label_folder + '/' + pathlib.Path(label).name
        label_obj.rename(label_target)
        # print(label_obj.as_posix(), label_obj.rename(label_target))


def merge_iterate(pathstr1, pathstr2):
    """Recurrently iterate both two folders.

    Returns:
        (list, list): two folders' subitems.
    """
    path1 = pathlib.Path(pathstr1)
    path2 = pathlib.Path(pathstr2)
    subitem1 = []
    subitem2 = []
    for p, q in zip(path1.rglob('*'), path2.rglob('*')):
        subitem1.append(p.as_posix())
        subitem2.append(q.as_posix())
    sorted_subitem1 = sorted(subitem1)
    sorted_subitem2 = sorted(subitem2)
    return sorted_subitem1, sorted_subitem2


def json_generate(train_val_path, test_path, patterns=['* T*.gz', '*-T*.gz', '*T1*.gz']):
    """生成数据集的json文件

    Args:
        train_val_path (str): 训练集、验证集的路径. e.g, '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211214/shidaoai/sichuan'
        test_path (str): 测试集的路径. e.g, '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211214/shidaoai/beijing'
    """
    # 处理四川数据，随机8 2分，作为训练集和验证集。
    path = pathlib.Path(train_val_path)

    data = []

    for pattern in patterns:
        data.extend({'image': list(item.parent.rglob('*CT*.gz'))[0].as_posix(
        ), 'label': item.as_posix()} for item in list(path.rglob(pattern)) if 'LACKT' not in item.name)

    data_train = sample(data, int(len(data) * 0.8))
    data_val = [item for item in data if item not in data_train]

    json_dict = {'training': data_train, 'validation': data_val}

    # 处理北京数据，作为测试集。

    path_test = pathlib.Path(test_path)

    json_dict['test'] = []

    for pattern in patterns:
        json_dict['test'].extend({'image': list(item.parent.rglob('*CT*.gz'))[0].as_posix(), 'label': item.as_posix()}
                                 for item in list(path_test.rglob(pattern)) if 'LACKT' not in item.name)

    json_file = json.dumps(json_dict, indent=4, sort_keys=False)

    # 查看训练集、验证集、测试集的数据量，并保存json文件。

    print(len(json_dict['training']), len(
        json_dict['validation']), len(json_dict['test']))

    with open('dataset.json', 'w') as f:
        f.write(json_file)


def check_contrast(datastr):
    datapath = pathlib.Path(datastr)
    imgs = datapath.rglob(pattern='*CT*')

    with open('check_contrast.txt', 'w') as f:
        for img in imgs:
            ct = sitk.ReadImage(img.as_posix())
            ct_array = sitk.GetArrayFromImage(ct)
            print(ct_array.min(), ct_array.max(), img)
            f.write(f'{ct_array.min()}, {ct_array.max()}, {img}\n')
            f.flush()
