#!/usr/bin/env python3


import pathlib
import os
import json
from random import sample
import sys
import SimpleITK as sitk
import numpy as np
import nibabel as nib


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


def json_generate(train_val_path, test_path, mask_patterns=['* T*.gz', '*-T*.gz', '*T1*.gz'], img_pattern='*CT*.gz'):
    """生成数据集的json文件

    Args:
        train_val_path (str): 训练集、验证集的路径. e.g, '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211214/shidaoai/sichuan'
        test_path (str): 测试集的路径. e.g, '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211214/shidaoai/beijing'
    """
    # 处理四川数据，随机8 2分，作为训练集和验证集。
    path = pathlib.Path(train_val_path)

    data = []

    for pattern in mask_patterns:
        data.extend({'image': list(item.parent.rglob(img_pattern))[0].as_posix(
        ), 'label': item.as_posix()} for item in list(path.rglob(pattern)) if 'LACKT' not in item.name)

    data_train = sample(data, int(len(data) * 0.8))
    data_val = [item for item in data if item not in data_train]

    json_dict = {'training': data_train, 'validation': data_val}

    # 处理北京数据，作为测试集。

    path_test = pathlib.Path(test_path)

    json_dict['test'] = []

    for pattern in patterns:
        json_dict['test'].extend({'image': list(item.parent.rglob(img_pattern))[0].as_posix(), 'label': item.as_posix()}
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

def no_label():
    from monai.transforms import (
        AddChanneld,
        Compose,
        LoadImaged,
        ToTensord,
    )

    from monai.data import (
        DataLoader,
        CacheDataset,
        load_decathlon_datalist,
    )

    import pathlib

    orig_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            ToTensord(keys=['image', 'label'])
        ]
    )

    split_JSON = "dataset.json"
    datasets = split_JSON

    train_files = load_decathlon_datalist(datasets, True, "all")

    orig_ds = CacheDataset(
        data=train_files,
        transform=orig_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )

    # 未做变换的训练集
    orig_loader = DataLoader(orig_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # no_label = [item['image_meta_dict']['filename_or_obj'][0] for item in orig_loader if item['label'].sum() == 0 ]

    with open(pathlib.Path('no_label.txt').as_posix(), 'w') as f:
        for item in orig_loader:
            if item['label'].sum() == 0:
                print(item['label_meta_dict']['filename_or_obj'][0])
                f.write(item['label_meta_dict']['filename_or_obj'][0] + '\n')
                f.flush()

    # with open(pathlib.Path('dataset', 'no_label.txt').as_posix(), 'w') as f:
    #     for item in no_label:
    #         f.write(item + '\n')



def get_roi(img_path, mask_path, save_root, save_folder_name='cropped', radius=None):
    # Get save path.
    img_path = pathlib.Path(img_path)
    mask_path = pathlib.Path(mask_path)
    save_root = pathlib.Path(save_root)

    img_relative = img_path.relative_to(save_root)
    mask_relative = mask_path.relative_to(save_root)

    img_savename = pathlib.Path(save_root.as_posix() + os.sep + save_folder_name + os.sep + img_relative.as_posix())
    mask_savename = pathlib.Path(save_root.as_posix() + os.sep + save_folder_name + os.sep + mask_relative.as_posix())

    print(f'{img_relative} | {mask_relative}')

    # Load img and mask.
    img = nib.load(img_path.as_posix())
    img_array = img.get_fdata()
    img_shape = img_array.shape

    mask = nib.load(mask_path.as_posix())
    mask_array = mask.get_fdata()
    mask_shape = mask_array.shape

    for i in range(len(img_shape)):
        if img_shape[i] != mask_shape[i]:
            # Img does not match with mask. They are in different shape.
            return -1

    nozero = np.nonzero(mask_array)

    try:
        # The first dimension: x
        # The second dimension: y
        # The third dimension: z
        x_min = nozero[0].min()
        x_max = nozero[0].max()
        y_min = nozero[1].min()
        y_max = nozero[1].max()
        z_min = nozero[2].min()
        z_max = nozero[2].max()
    except:
        # No mask.
        return -3

    x_width = x_max - x_min + 1
    y_width = y_max - y_min + 1
    z_width = z_max - z_min + 1

    x_radius = x_width // 2
    y_radius = y_width // 2
    z_radius = z_width // 2
    max_radius = max(x_radius, y_radius, z_radius) if not radius else radius

    x_center = x_min + x_radius
    y_center = y_min + y_radius
    z_center = z_min + z_radius

    x_roi_min_orig = x_center - (max_radius + 1)
    x_roi_max_orig = x_center + (max_radius + 1)
    y_roi_min_orig = y_center - (max_radius + 1)
    y_roi_max_orig = y_center + (max_radius + 1)
    z_roi_min_orig = z_center - (max_radius + 1)
    z_roi_max_orig = z_center + (max_radius + 1)

    # Note: If the voxal concatinates with the image boundary, there will be a bug that the cropped array will not be a cube. It need to be fixed in the future.
    x_roi_min_norm = max(x_roi_min_orig, 0)
    x_roi_max_norm = min(x_roi_max_orig, img_shape[0])
    y_roi_min_norm = max(y_roi_min_orig, 0)
    y_roi_max_norm = min(y_roi_max_orig, img_shape[1])
    z_roi_min_norm = max(z_roi_min_orig, 0)
    z_roi_max_norm = min(z_roi_max_orig, img_shape[2])

    mask_roi_array = mask_array[x_roi_min_norm:x_roi_max_norm, y_roi_min_norm:y_roi_max_norm, z_roi_min_norm:z_roi_max_norm]
    img_roi_array = img_array[x_roi_min_norm:x_roi_max_norm, y_roi_min_norm:y_roi_max_norm, z_roi_min_norm:z_roi_max_norm]

    # Image intensity limitation and normalization.
    img_roi_array = _scale_intensity(img_roi_array)

    # print(img.shape)
    # print(nozero)
    # print(x_min, x_max, y_min, y_max, z_min, z_max)
    # print(x_center, y_center, z_center)
    # print(x_width, y_width, z_width)
    # print(x_radius, y_radius, z_radius)
    # print(max_radius)
    # print(x_roi_min_orig, x_roi_max_orig, y_roi_min_orig, y_roi_max_orig, z_roi_min_orig, z_roi_max_orig)
    # print(x_roi_min_norm, x_roi_max_norm, y_roi_min_norm, y_roi_max_norm, z_roi_min_norm, z_roi_max_norm)
    # print(mask_roi_array.shape)

    # if mask_roi_array.shape[0] != mask_roi_array.shape[1] or mask_roi_array.shape[0] != mask_roi_array.shape[2]:
    #     # The cropped area is not a cube. Error generated, return.
    #     return -2

    img_savefolder = img_savename.parent
    mask_savefolder = mask_savename.parent

    if not img_savefolder.exists():
        img_savefolder.mkdir(parents=True, exist_ok=True)
    if not mask_savefolder.exists():
        mask_savefolder.mkdir(parents=True, exist_ok=True)

    img_out = nib.Nifti1Image(img_roi_array, affine=np.eye(4))
    # Get image header for fix resolution.
    img_out.header.set_zooms(img.header.get_zooms())
    img_out.to_filename(img_savename.as_posix())

    mask_out = nib.Nifti1Image(mask_roi_array, affine=np.eye(4))
    # Get mask header for fix resolution.
    mask_out.header.set_zooms(mask.header.get_zooms())
    mask_out.to_filename(mask_savename.as_posix())

    return max_radius


def _scale_intensity(img_array, a_min=0, a_max=1500, b_min=0, b_max=1):
    # Image intensity limitation.
    img_array[img_array > a_max] = 0

    # Image intensity normalization.
    img_array = (img_array - a_min) / (a_max - a_min)
    img_array = img_array * (b_max - b_min) + b_min

    return img_array

def scale_intensity(img_path, save_root, a_min=0, a_max=1500, b_min=0, b_max=1, save_folder_name='scale_intensity_single'):
    img_path = pathlib.Path(img_path)
    save_root = pathlib.Path(save_root)

    # Get image save path.
    img_relative = img_path.relative_to(save_root)
    img_savename = pathlib.Path(save_root.as_posix() + os.sep + save_folder_name + os.sep + img_relative.as_posix())
    print(f'{img_relative}')

    # Load image.
    img = nib.load(img_path.as_posix())
    img_array = img.get_fdata()
    img_shape = img_array.shape


    # Image intensity limitation.
    img_array[img_array > a_max] = 0
    # Image intensity normalization.
    img_array = (img_array - a_min) / (a_max - a_min)
    img_array = img_array * (b_max - b_min) + b_min

    # Save image.
    img_savefolder = img_savename.parent
    if not img_savefolder.exists():
        img_savefolder.mkdir(parents=True, exist_ok=True)

    img_out = nib.Nifti1Image(img_array, affine=np.eye(4))
    # Get image header for fix resolution.
    img_out.header.set_zooms(img.header.get_zooms())
    img_out.to_filename(img_savename.as_posix())


def peek_image_intensity(img_path):
    print(img_path)

    img_path = pathlib.Path(img_path)
    img = nib.load(img_path.as_posix())
    return img.header.get_zooms()