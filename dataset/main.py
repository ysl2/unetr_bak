import datatools as dt
import json
import SimpleITK as sitk
import pathlib
import os
import sys
import nibabel as nib
import copy


database_linux = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/'
database_windows = 'F:\\shidaoai'

output_linux = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/unetr_output'


def backup():
    # dt.check_contrast('F:\\shidaoai')
    # dt.json_generate('F:\\shidaoai\\sichuan', 'F:\\shidaoai\\beijing')

    # dt.check_contrast('/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/')
    # dt.json_generate('/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/sichuan', '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/beijing')

    # target_files =  dt.get_targets(pathstr='/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/unetr_output', patterns=['**/99/**/*_pred.nii.gz'])
    # target_length = len(target_files)
    # target_files = json.dumps(target_files, indent=4)
    # # pred predict image label
    #
    # print(target_files)
    # print(target_length)
    pass


def get_roi_total():
    save_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai'

    file = open('dataset/dataset.json', 'r')
    dataset = json.load(file)

    tags = ['training', 'validation', 'test']

    max_radius = -1
    min_radius = 9999
    with open('crop_log.txt', 'w') as f:
        for tag in tags:
            for i in range(len(dataset[tag])):
                img_path = dataset[tag][i]['image']
                mask_path = dataset[tag][i]['label']
                return_value = dt.get_roi(
                    img_path, mask_path, save_root, radius=63)
                if return_value < 0:
                    f.write(f'{return_value} | {img_path}\n')
                    f.flush()
                if return_value > max_radius:
                    max_radius = return_value
                    print(f'Current max radius: {max_radius}')
                if 0 < return_value < min_radius:
                    min_radius = return_value
                    print(f'Current min radius: {min_radius}')
        print(f'Total max radius: {max_radius}')
        print(f'Total min radius: {min_radius}')
        f.write(f'Total max radius: {max_radius}\n')
        f.write(f'Total min radius: {min_radius}\n')
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


def find_CT_prediction_pair_from_json_validation():
    predict_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/unetr_output/24'
    f = open('dataset/dataset.json')
    validation = json.load(f)['validation']

    save_root = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai'

    counter = 0
    with open('crop_log.txt', 'w') as f:
        for item in validation:
            img_path = pathlib.Path(item['image'])
            img_name = img_path.name
            img_number = img_name.split('_')[0]

            pred_name = img_number + os.sep + img_number + '_pred.nii.gz'
            pred_str = predict_root + os.sep + pred_name
            return_value = dt.get_roi(
                img_path, pred_str, save_root, save_folder_name='croppped_img_mask_pred', radius=63)
            counter += 1
            if return_value < 0:
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
                element_name = element_number + '.nii.gz'
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


if __name__ == '__main__':
    # get_roi_total()
    # get_roi_single()
    # test_scale_intensity()
    # iter_intensity()
    # test_move_data()
    # find_CT_prediction_pair_from_json_validation()
    # test_check_pixel()
    copy_training_cropped_from_json()