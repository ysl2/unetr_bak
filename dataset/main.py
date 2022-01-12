import imgtools as it
import json
import SimpleITK as sitk
import pathlib
import os
import sys
import nibabel as nib
import copy
import numpy as np
import shutil

database_linux = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/'
database_windows = 'F:\\shidaoai'



def test_generate_json():
    train_val_path = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/sichuan'
    test_path = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/beijing'
    it.generate_json(train_val_path, test_path, json_savepath='dataset/json/dataset.json', mask_patterns=['*T[_1 ]*.gz'])
    json_path = 'dataset/json/dataset.json'
    it.generate_convert_json_from_json(json_path)


def test_generate_convert_json_from_json():
    # json_path='dataset/json/dataset.json'
    json_path='dataset/json/dataset_unetr_1332_332_264.json'
    it.generate_convert_json_from_json(json_path)


def test_get_roi_total():

    file = open('dataset/json/dataset_unetr_1332_332_264_convert.json', 'r')
    
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
            # for i in range(len(dataset['validation'])):
                # 0: gt
                # 1: cropped_shidaoai
                # 3: cropped_img_mask/img
                # 4: spacial_input
                # 5: spacial_output/img
                # 7: unetr_output/24
                # 8: cropped_shidaoai_no_spacial
                # 9: shidaoai_spacial
                img_path = dataset[tag][i]['image'][0]
                mask_path = dataset[tag][i]['label'][0]
                img_savepath = dataset[tag][i]['image'][4]
                mask_savepath = dataset[tag][i]['label'][4]

                # mask_predpath = dataset[tag][i]['label'][7]
                # return_value = it.get_roi(img_path, mask_path, img_savepath, mask_savepath, x_area=48, y_area=61, z_area=28, pred_path=mask_predpath)
                return_value = it.get_roi(img_path, mask_path, img_savepath, mask_savepath, x_area=48, y_area=61, z_area=28)

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


def test_get_roi_single():
    # save_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai'
    # # for item in pathlib.Path(database_linux).rglob('**/*298225*GTV-T*.nii.gz'):
    # for item in pathlib.Path(database_linux).rglob('**/*GTV-T*.nii.gz'):
    #     image = pathlib.Path(item.parent.as_posix() +
    #                          os.sep + item.name.split('_')[0] + '_CT.nii.gz')
    #     return_value = it.get_roi(image.as_posix(), item.as_posix(
    #     ), save_root=save_root, save_folder_name='cropped_single', radius=63)
    #     print(return_value)
    #     sys.exit(0)

    img_orig1 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_output/img/sichuan_liutong_240124.nii.gz'
    img_target1 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/temp_trash/img/sichuan_liutong_240124.nii.gz'
    mask_orig1 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_output/mask/sichuan_liutong_240124.nii.gz'
    mask_target1 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/temp_trash/mask/sichuan_liutong_240124.nii.gz'

    img_orig2 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_output/img/sichuan_liutong_121661.nii.gz'
    img_target2 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/temp_trash/img/sichuan_liutong_121661.nii.gz'
    mask_orig2 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_output/mask/sichuan_liutong_121661.nii.gz'
    mask_target2 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/temp_trash/mask/sichuan_liutong_121661.nii.gz'

    img_orig3 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_output/img/sichuan_liutong_305703.nii.gz'
    img_target3 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/temp_trash/img/sichuan_liutong_305703.nii.gz'
    mask_orig3 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_output/mask/sichuan_liutong_305703.nii.gz'
    mask_target3 = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/temp_trash/mask/sichuan_liutong_305703.nii.gz'

    it.get_roi(img_orig1, mask_orig1, img_target1, mask_target1, x_area=48, y_area=61, z_area=28)
    it.get_roi(img_orig2, mask_orig2, img_target2, mask_target2, x_area=48, y_area=61, z_area=28)
    it.get_roi(img_orig3, mask_orig3, img_target3, mask_target3, x_area=48, y_area=61, z_area=28)


def test_scale_intensity():
    file = open('dataset/json/dataset_unetr_1332_332_264_convert.json', 'r')
    
    dataset = json.load(file)

    tags = ['training', 'validation', 'test']

    with open('logs/crop_log.txt', 'w') as f:
        for tag in tags:
            for i in range(len(dataset[tag])):
            # for i in range(len(dataset['validation'])):
                # 0: gt
                # 1: cropped_shidaoai
                # 3: cropped_img_mask/img
                # 4: spacial_input
                # 5: spacial_output/img
                # 7: unetr_output/24
                # 8: cropped_shidaoai_no_spacial
                # 9: shidaoai_spacial
                # 10: shidaoai_spacial_scale_intensity
                img_path = dataset[tag][i]['image'][9]
                img_savepath = dataset[tag][i]['image'][10]
                it.scale_intensity(img_path, img_savepath)



def test_check_zooms():
    with open('dataset/json/dataset_convert.json', 'r') as f:
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
            zooms = it.check_zooms(img_path)
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


def test_check_pixel_unetr_pred():
    data_path = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/unetr_output'
    i = 4
    min_epoch = []
    max_epoch = []
    min_missing = None
    max_missing = None
    while i <= 99:
        data_path_pattern = '**/' + str(i) + '/**/*_pred.nii.gz'
        missing = it.check_pixel(
            data_path, data_path_pattern=data_path_pattern)

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


def test_check_pixel_2D_UNet():
    data_path = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/cropped_img_mask_pred/out'
    return_value = it.check_pixel(data_path, data_path_pattern='*.nii.gz')
    print('-' * 10)
    print('No pixel:')
    for item in return_value:
        print(item)



def test_json_move():
    # 0: gt
    # 1: cropped_shidaoai
    # 3: cropped_img_mask/img
    # 4: spacial_input
    # 5: spacial_output/img
    # 7: unetr_output/24
    # 8: cropped_shidaoai_no_spacial
    # 9: shidaoai_spacial
    convert_json_path = 'dataset/json/dataset_unetr_1332_332_264_convert.json'
    it.json_move(convert_json_path=convert_json_path, tags=['training', 'validation', 'test'], input_index=9, output_index=10, mode='copy', log_path='logs/json_move.txt')
    # it.json_move(convert_json_path=convert_json_path, tags=['validation'], input_index=7, output_index=4, mode='copy', log_path='logs/json_move.txt')


# def test_get_difference_between_json():
#     json_path1 = 'dataset/json/dataset.json'
#     json_path2 = 'dataset/json/dataset_unetr_1332_334_276.json'
#     log_path = 'logs/difference_json.txt'
#     it.get_difference_between_json(json_path1, json_path2, log_path=log_path)


# def get_gt_pred():
#     trash = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/temp_trash'
#     dataset = json.load(open('dataset/json/dataset_unetr_1332_332_264_convert.json'))
#     gt_img_path = dataset['validation'][0]['image'][0]
#     gt_mask_path = dataset['validation'][0]['label'][0]
#     pred_mask_path = dataset['validation'][0]['label'][7]

#     shutil.copy(gt_img_path, trash + os.sep + 'gt_img.nii.gz')
#     shutil.copy(gt_mask_path, trash + os.sep + 'gt_mask.nii.gz')
#     shutil.copy(pred_mask_path, trash + os.sep + 'pred_mask.nii.gz')


if __name__ == '__main__':
    # test_json_move()
    # test_generate_convert_json_from_json()
    # test_scale_intensity()
    test_check_pixel_2D_UNet()