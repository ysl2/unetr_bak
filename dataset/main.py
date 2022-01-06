import datatools as dt
import json
import SimpleITK as sitk
import pathlib
import os
import sys
import nibabel as nib


database_linux = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/shidaoai/'
database_windows = 'F:\\shidaoai'

output_linux = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/unetr_output'

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

# for item in pathlib.Path(output_linux).rglob('**/99/**/*_pred.nii.gz'):
    # image = sitk.ReadImage(item.as_posix(), imageIO="PNGImageIO")
    # image = sitk.ReadImage(item.as_posix())
    # image_array = sitk.GetArrayViewFromImage(image)
    # image_shape = image.GetSize()
    # print(image_shape)
    # if image_array.sum() <= 0:
    #     print(item.as_posix())




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
                return_value = dt.get_roi(img_path, mask_path, save_root, radius=63)
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
        image = pathlib.Path(item.parent.as_posix() + os.sep + item.name.split('_')[0] + '_CT.nii.gz')
        return_value = dt.get_roi(image.as_posix(), item.as_posix(), save_root=save_root, save_folder_name='cropped_single', radius=63)
        print(return_value)
        sys.exit(0)

if __name__ == '__main__':
    get_roi_total()
    # get_roi_single()