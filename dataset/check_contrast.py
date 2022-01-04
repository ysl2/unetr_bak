import SimpleITK as sitk
import numpy as np
import pathlib
import sys

datastr = 'F:\\shidaoai'
datapath = pathlib.Path(datastr)
imgs = datapath.rglob(pattern='*CT*')

with open('check_contrast.txt', 'w') as f:
    for img in imgs:
        ct = sitk.ReadImage(img.as_posix())
        ct_array = sitk.GetArrayFromImage(ct)
        print(ct_array.min(), ct_array.max(), img)
        f.write(f'{ct_array.min()}, {ct_array.max()}, {img}\n')
        f.flush()

# sys.exit()

# def recur(path):
#     if path.is_file() and path.as_posix().contains('CT'):
#         ct = sitk.ReadImage(path.as_posix())
#         ct_array = sitk.GetArrayFromImage(ct)
#         print(ct_array.min(), ct_array.max(), path)
#         return
#     for p in path.iterdir():
#         recur(p)

# if __name__ == '__main__':
#     recur(datapath)