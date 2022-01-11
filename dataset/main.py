import datatools as dt
import json
import SimpleITK as sitk
import pathlib
import os
import sys
import nibabel as nib
import copy
import numpy as np

database_linux = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/'
database_windows = 'F:\\shidaoai'

output_linux = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/unetr_output'


def test_json_generate():
    train_val_path = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/sichuan'
    test_path = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/beijing'
    dt.json_generate(train_val_path, test_path, json_savepath='dataset/dataset.json', mask_patterns=['*T[_1 ]*.gz'])
    json_path = 'dataset/dataset.json'
    dt.generate_convert_json_from_json(json_path)


def test_generate_convert_json_from_json():
    # json_path='dataset/dataset.json'
    json_path='dataset/dataset_unetr_1332_334_276.json'
    dt.generate_convert_json_from_json(json_path)


def test_get_roi_total():
    # save_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai'

    file = open('dataset/dataset_convert.json', 'r')
    # file = open('dataset/dataset_test.json', 'r')
    dataset = json.load(file)

    tags = ['training', 'validation', 'test']

    with open('logs/crop_log.txt', 'w') as f:
        for tag in tags:

            max_radius = -1
            max_radius_img = None

            min_radius = 9999
            min_radius_img = None

            max_x_radius = -1
            max_x_radius_img = None

            min_x_radius = 9999
            min_x_radius_img = None

            max_y_radius = -1
            max_y_radius_img = None

            min_y_radius = 9999
            min_y_radius_img = None

            max_z_radius = -1
            max_z_radius_img = None

            min_z_radius = 9999
            min_z_radius_img = None

            for i in range(len(dataset[tag])):
                img_path = dataset[tag][i]['image'][5]
                mask_path = dataset[tag][i]['label'][5]
                img_savepath = dataset[tag][i]['image'][3]
                mask_savepath = dataset[tag][i]['label'][3]
                return_value = dt.get_roi(img_path, mask_path, img_savepath, mask_savepath)
                if isinstance(return_value, np.int64) or isinstance(return_value, int):
                    if return_value < 0:
                        f.write(f'{return_value} | {img_path}\n')
                        f.flush()
                    if return_value > max_radius:
                        max_radius = return_value
                        max_radius_img = img_path
                        print(f'{tag} | Current max radius: {max_radius}, in {max_radius_img}')
                    if 0 < return_value < min_radius:
                        min_radius = return_value
                        min_radius_img = img_path
                        print(f'{tag} | Current min radius: {min_radius}, in {min_radius_img}')
                elif isinstance(return_value, tuple):
                    if return_value[0] > max_x_radius:
                        max_x_radius = return_value[0]
                        max_x_radius_img = img_path
                        print(f'{tag} | Current max x radius: {max_x_radius}, in {max_x_radius_img}')
                    if return_value[0] < min_x_radius:
                        min_x_radius = return_value[0]
                        min_x_radius_img = img_path
                        print(f'{tag} | Current min x radius: {min_x_radius}, in {min_x_radius_img}')
                    if return_value[1] > max_y_radius:
                        max_y_radius = return_value[1]
                        max_y_radius_img = img_path
                        print(f'{tag} | Current max y radius: {max_y_radius}, in {max_y_radius_img}')
                    if return_value[1] < min_y_radius:
                        min_y_radius = return_value[1]
                        min_y_radius_img = img_path
                        print(f'{tag} | Current min y radius: {min_y_radius}, in {min_y_radius_img}')
                    if return_value[2] > max_z_radius:
                        max_z_radius = return_value[2]
                        max_z_radius_img = img_path
                        print(f'{tag} | Current max z radius: {max_z_radius}, in {max_z_radius_img}')
                    if return_value[2] < min_z_radius:
                        min_z_radius = return_value[2]
                        min_z_radius_img = img_path
                        print(f'{tag} | Current min z radius: {min_z_radius}, in {min_z_radius_img}')

            if isinstance(return_value, np.int64) or isinstance(return_value, int):
                print(f'{tag} | Total max radius: {max_radius}, in {max_radius_img}')
                print(f'{tag} | Total min radius: {min_radius}, in {min_radius_img}')
                f.write(f'{tag} | Total max radius: {max_radius}, in {max_radius_img}\n')
                f.write(f'{tag} | Total min radius: {min_radius}, in {min_radius_img}\n')
                f.flush()
            elif isinstance(return_value, tuple):
                print(f'{tag} | Total max x radius: {max_x_radius}, in {max_x_radius_img}')
                print(f'{tag} | Total min x radius: {min_x_radius}, in {min_x_radius_img}')
                print(f'{tag} | Total max y radius: {max_y_radius}, in {max_y_radius_img}')
                print(f'{tag} | Total min y radius: {min_y_radius}, in {min_y_radius_img}')
                print(f'{tag} | Total max z radius: {max_z_radius}, in {max_z_radius_img}')
                print(f'{tag} | Total min z radius: {min_z_radius}, in {min_z_radius_img}')
                f.write(f'{tag} | Total max x radius: {max_x_radius}, in {max_x_radius_img}\n')
                f.write(f'{tag} | Total min x radius: {min_x_radius}, in {min_x_radius_img}\n')
                f.write(f'{tag} | Total max y radius: {max_y_radius}, in {max_y_radius_img}\n')
                f.write(f'{tag} | Total min y radius: {min_y_radius}, in {min_y_radius_img}\n')
                f.write(f'{tag} | Total max z radius: {max_z_radius}, in {max_z_radius_img}\n')
                f.write(f'{tag} | Total min z radius: {min_z_radius}, in {min_z_radius_img}\n')
                f.flush()


def get_roi_single():
    save_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai'
    # for item in pathlib.Path(database_linux).rglob('**/*298225*GTV-T*.nii.gz'):
    for item in pathlib.Path(database_linux).rglob('**/*GTV-T*.nii.gz'):
        image = pathlib.Path(item.parent.as_posix() +
                             os.sep + item.name.split('_')[0] + '_CT.nii.gz')
        return_value = dt.get_roi(image.as_posix(), item.as_posix(
        ), save_root=save_root, save_folder_name='cropped_single', radius=63)
        print(return_value)
        sys.exit(0)


def test_scale_intensity():
    save_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai'
    # for item in pathlib.Path(database_linux).rglob('**/*298225*GTV-T*.nii.gz'):
    for item in pathlib.Path(database_linux).rglob('**/*GTV-T*.nii.gz'):
        image = pathlib.Path(item.parent.as_posix() +
                             os.sep + item.name.split('_')[0] + '_CT.nii.gz')
        dt.scale_intensity(image.as_posix(), save_root=save_root)
        sys.exit(0)


def iter_intensity():
    save_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai'

    file = open('dataset/dataset.json', 'r')
    dataset = json.load(file)

    tags = ['training', 'validation', 'test']

    intensity_x = None
    intensity_y = None
    intensity_z = None
    record = None

    for tag in tags:
        for i in range(len(dataset[tag])):
            img_path = dataset[tag][i]['image']
            mask_path = dataset[tag][i]['label']
            return_value = dt.peek_image_intensity(img_path)
            if intensity_x == None:
                intensity_x = return_value[0]
                intensity_y = return_value[1]
                intensity_z = return_value[2]
                record = img_path
            elif intensity_x != return_value[0]:
                print('Got not equal intensity.')
                print(f'{record} | ({intensity_x}, {intensity_y}, {intensity_z})')
                print(f'{img_path} | {return_value}')
                return


def test_check_zooms():
    with open('dataset/dataset_convert.json', 'r') as f:
        dataset = json.load(f)

    max_x_zoom = -1
    min_x_zoom = 9999
    max_y_zoom = -1
    min_y_zoom = 9999
    max_z_zoom = -1
    min_z_zoom = 9999

    tags = ['training', 'validation', 'test']
    i = 0
    for tag in tags: # ['training', 'validation', 'test']
        for item in dataset[tag]: # item: {'image': [], 'label': []}
            img_path = item['image'][4]
            zooms = dt.check_zooms(img_path)
            i += 1
            max_x_zoom = max(max_x_zoom, zooms[0])
            min_x_zoom = min(min_x_zoom, zooms[0])
            max_y_zoom = max(max_y_zoom, zooms[1])
            min_y_zoom = min(min_y_zoom, zooms[1])
            max_z_zoom = max(max_z_zoom, zooms[2])
            min_z_zoom = min(min_z_zoom, zooms[2])
    print(f'Max x zoom: {max_x_zoom}, min x zoom: {min_x_zoom}')
    print(f'Max y zoom: {max_y_zoom}, min y zoom: {min_y_zoom}')
    print(f'Max z zoom: {max_z_zoom}, min z zoom: {min_z_zoom}')
    print(f'Total: {i}')


def find_CT_prediction_pair_from_json_validation():
    predict_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/unetr_output/24'
    f = open('dataset/dataset.json')
    validation = json.load(f)['validation']

    save_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai'

    counter = 0
    with open('logs/crop_log.txt', 'w') as f:
        for item in validation:
            img_path = pathlib.Path(item['image'])
            img_name = img_path.name
            img_number = img_name.split('_')[0]

            pred_name = img_number + os.sep + img_number + '_pred.nii.gz'
            pred_str = predict_root + os.sep + pred_name
            return_value = dt.get_roi(
                img_path, pred_str, save_root, save_folder_name='croppped_img_mask_pred', radius=63)
            counter += 1
            if return_value and return_value < 0:
                f.write(f'{return_value} | {pred_str}\n')
                f.flush()
        print(f'Total processed img: {counter}.')
        f.write(f'Total processed img: {counter}.')
        f.flush()


def test_check_pixel():
    i = 4
    min_epoch = []
    max_epoch = []
    min_missing = None
    max_missing = None
    while i <= 99:
        data_path_pattern = '**/' + str(i) + '/**/*_pred.nii.gz'
        missing = dt.check_pixel(
            output_linux, data_path_pattern=data_path_pattern)

        if not min_missing:
            min_missing = len(missing)
            min_epoch.append(i)
        elif len(missing) == min_missing:
            min_epoch.append(i)
        elif len(missing) < min_missing:
            min_missing = len(missing)
            min_epoch = [i]

        if not max_missing:
            max_missing = len(missing)
            max_epoch.append(i)
        elif len(missing) == max_missing:
            max_epoch.append(i)
        elif len(missing) > max_missing:
            max_missing = len(missing)
            max_epoch = [i]
        i += 5

    print(
        f'Max missing epochs: {max_epoch}, number of max missing: {max_missing}')
    print(
        f'Min missing epochs: {min_epoch}, number of min missing: {min_missing}')


def copy_training_cropped_from_json():
    # You need to know two path:
    # 1. img_location, mask_location
    # 2. img_target, mask_target

    # Define save root.
    data_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai'
    data_location_folder = 'cropped'
    data_target_folder = 'cropped_img_mask_train'
    data_path = data_root + os.sep + data_target_folder

    # Define img and mask save path.
    img_savepath = data_path + os.sep + 'img'
    img_savepath = pathlib.Path(img_savepath)
    mask_savepath = data_path + os.sep + 'mask'
    mask_savepath = pathlib.Path(mask_savepath)

    # Load list from json.
    f = open('dataset/dataset.json')
    training = json.load(f)['training']
    total = len(training)

    i = 1
    for item in training:
        # Get original img and mask path.
        img_path = item['image']
        img_path = pathlib.Path(img_path)
        img_name = img_path.name

        mask_path = item['label']
        mask_path = pathlib.Path(mask_path)
        mask_name = mask_path.name

        # Set img and mask location path.
        img_relative = img_path.relative_to(data_root)
        img_location = data_root + os.sep + \
            data_location_folder + os.sep + img_relative.as_posix()
        img_location = pathlib.Path(img_location)

        mask_relative = mask_path.relative_to(data_root)
        mask_location = data_root + os.sep + \
            data_location_folder + os.sep + mask_relative.as_posix()
        mask_location = pathlib.Path(mask_location)

        # Set img and mask target path.
        img_number = img_name.split('_')[0]
        img_new_name = img_number + '.nii.gz'
        img_target = img_savepath.as_posix() + os.sep + img_new_name
        img_target = pathlib.Path(img_target)

        mask_number = mask_name.split('_')[0]
        mask_new_name = mask_number + '.nii.gz'
        mask_target = mask_savepath.as_posix() + os.sep + mask_new_name
        mask_target = pathlib.Path(mask_target)

        # Create savepath folders.
        if not img_savepath.exists():
            img_savepath.mkdir(parents=True, exist_ok=True)
        if not mask_savepath.exists():
            mask_savepath.mkdir(parents=True, exist_ok=True)

        # Judge if there are already the same name. If yes, rename it.
        j = 0
        union = [img_target, mask_target]
        while True:
            if not union[0].exists():
                break
            for k in range(len(union)):
                element = union[k]
                element_name = element.name
                element_parent = element.parent.as_posix()
                element_number = element_name.split('.')[0]
                element_number = element_number.split('_')[0]
                element_number = element_number + '_' + str(j)
                element_name = element_number + '_MASK.nii.gz'
                element = pathlib.Path(element_parent + os.sep + element_name)
                union[k] = element
            print(union[0])
            print(union[1])
            j += 1
        # Copy contents.
        union[0].write_bytes(img_location.read_bytes())
        union[1].write_bytes(mask_location.read_bytes())

        print(f'{i}/{total}: {img_number}')
        i += 1


def test_json_move():
    convert_json_path = 'dataset/dataset_convert.json'
    dt.json_move(convert_json_path=convert_json_path, tags=['training', 'validation', 'test'], input_index=6, output_index=4, mode='cut', log_path='logs/json_move.txt')


def test_get_difference_between_json():
    json_path1 = 'dataset/dataset.json'
    json_path2 = 'dataset/dataset_unetr_1332_334_276.json'
    log_path = 'logs/difference_json.txt'
    dt.get_difference_between_json(json_path1, json_path2, log_path=log_path)

if __name__ == '__main__':
    # test_json_generate()
    # test_get_roi_total()
    # test_get_roi_single()
    # test_scale_intensity()
    # iter_intensity()
    # test_move_data()
    # find_CT_prediction_pair_from_json_validation()
    # test_check_pixel()
    # copy_training_cropped_from_json()
    # test_generate_convert_json_from_json()
    # test_json_move()
    # test_check_zooms()
    test_get_difference_between_json()
