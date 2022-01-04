#!/usr/bin/env python3


import pathlib
import json
from random import sample
import sys
import SimpleITK as sitk
import numpy as np
import nibabel as nib

# patterns = ['* T*.gz', '*-T*.gz', '*T1*.gz']


def get_pairs(pathstr, img_pattern, label_patterns):
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


def get_targets(pathstr, patterns):
    """
    Find some target files in a specific folder path.
    Args:
        pathstr (str): A specific folder path.
        patterns (list(str)): The patterns you want to find.
    Return:
        target_files (list): A list of target files.
    """
    path = pathlib.Path(pathstr)
    if patterns and isinstance(patterns, list):
        target_files = []
        for pattern in patterns:
            target_files.extend([item.as_posix()
                                for item in list(path.rglob(pattern))])
        return target_files

def get_roi(img_path, mask_path):
    img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayViewFromImage(img)
    img_shape = img_array.shape

    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayViewFromImage(mask)
    mask_shape = mask_array.shape

    for i in range(len(img_shape)):
        if img_shape[i] != mask_shape[i]:
            print('Img does not match with mask. They are in different shape')
            return

    nozero = np.nonzero(mask_array)

    # The first dimension: z
    # The second dimension: y
    # The third dimension: x
    x_min = nozero[2].min()
    x_max = nozero[2].max()
    y_min = nozero[1].min()
    y_max = nozero[1].max()
    z_min = nozero[0].min()
    z_max = nozero[0].max()

    x_width = x_max - x_min + 1
    y_width = y_max - y_min + 1
    z_width = z_max - z_min + 1

    x_radis = x_width // 2
    y_radis = y_width // 2
    z_radis = z_width // 2
    max_radis = max(x_radis, y_radis, z_radis)

    x_mid = x_min + x_radis
    y_mid = y_min + y_radis
    z_mid = z_min + z_radis

    # Note: If the voxal concatinates with the image boundary, there will be a bug that the cropped array will not be a cube. It need to be fixed in the future.
    x_roi_min = max(x_mid - max_radis - 1, 0)
    x_roi_max = min(x_mid + max_radis + 1, img_shape[2])
    y_roi_min = max(y_mid - max_radis - 1, 0)
    y_roi_max = min(y_mid + max_radis + 1, img_shape[1])
    z_roi_min = max(z_mid - max_radis - 1, 0)
    z_roi_max = min(z_mid + max_radis + 1, img_shape[0])

    mask_roi = mask[z_roi_min:z_roi_max, y_roi_min:y_roi_max, x_roi_min:x_roi_max]
    mask_roi_array = mask_array[z_roi_min:z_roi_max, y_roi_min:y_roi_max, x_roi_min:x_roi_max]
    img_roi = img[z_roi_min:z_roi_max, y_roi_min:y_roi_max, x_roi_min:x_roi_max]
    img_roi_array = img_array[z_roi_min:z_roi_max, y_roi_min:y_roi_max, x_roi_min:x_roi_max]

    print(x_min, x_max, y_min, y_max, z_min, z_max)
    print(x_width, y_width, z_width)
    print(x_radis, y_radis, z_radis)
    print(max_radis)
    print(x_mid, y_mid, z_mid)
    print(x_roi_min, x_roi_max, y_roi_min, y_roi_max, z_roi_min, z_roi_max)
    # mask_sum = np.sum(mask_array)
    # roi_sum = np.sum(mask_roi_array)
    # print(mask_sum, roi_sum)
    # sys.exit()

    img_out = nib.Nifti1Image(img_roi_array, affine=np.eye(4))
    img_out.header.get_xyzt_units()
    img_out.to_filename('test_img.nii.gz')

    mask_out = nib.Nifti1Image(mask_roi_array, affine=np.eye(4))
    mask_out.header.get_xyzt_units()
    mask_out.to_filename('test_mask.nii.gz')

    sys.exit()
