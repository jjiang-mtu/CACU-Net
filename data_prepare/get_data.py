import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

import copy
import skimage.measure as measure

ct_path = './dataset/AAATData/train/ct'
seg_path = './dataset/AAATData/train/label'

new_ct_path = './train/CT/'
new_seg_path = './train/GT/'

# Newly generated training data storage path
if os.path.exists('./train/') is True:
    shutil.rmtree('./train/')
os.mkdir('./train/')
os.mkdir(new_ct_path)
os.mkdir(new_seg_path)

upper = 1200
lower = -300
slice_thickness = 3
down_scale = 0.5
expand_slice = 20

file_index = 0  # Serial number for recording newly generated data
start_time = time()

def doCCO(Binary_Img):
    labeled_Img = measure.label(Binary_Img, connectivity=1)
    props = measure.regionprops(labeled_Img)
    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index  
    labeled_Img[labeled_Img != max_index] = 0
    labeled_Img[labeled_Img == max_index] = 1

    return labeled_Img
    
def CCO(SegImg):
    Retain_Img = np.zeros(np.shape(SegImg))

    # class 2
    Img_C2 = np.int8(SegImg == 2)
    Retain_C2 = doCCO(Img_C2)   
    Retain_Img[Retain_C2] = 2
    
    # class 1
    Img_C1 = np.int8(SegImg == 1)
    Retain_C1 = doCCO(Img_C1)
    Retain_Img[Retain_C1] = 1
    
    return Retain_Img 

for ct_file in os.listdir(ct_path):

    ct = sitk.ReadImage(os.path.join(ct_path, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(seg_path, ct_file.replace('volume', 'segmentation')), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # Truncate grayscale values outside of the threshold
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    mask_Seg = copy.deepcopy(seg_array)
    mask_Seg[mask_Seg > 0] = 1
    mask_Seg  = doCCO(mask_Seg)

    # Find the slice containing the aneurysm
    z = np.any(mask_Seg, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    # Expand slice in each direction
    if start_slice - expand_slice < 0:
        start_slice = 0
    else:
        start_slice -= expand_slice

    if end_slice + expand_slice >= mask_Seg.shape[0]:
        end_slice = mask_Seg.shape[0] - 1
    else:
        end_slice += expand_slice

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    print('file name:', ct_file)
    print('shape:', ct_array.shape)

    # Save data
    new_ct_array = ct_array
    new_seg_array = seg_array

    new_ct = sitk.GetImageFromArray(new_ct_array)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing(ct.GetSpacing())    
        

    new_seg = sitk.GetImageFromArray(new_seg_array)

    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing(ct.GetSpacing()) 

    new_ct_name = 'img-' + str(file_index) + '.nii.gz'
    new_seg_name = 'label-' + str(file_index) + '.nii.gz'

    sitk.WriteImage(new_ct, os.path.join(new_ct_path, new_ct_name))
    sitk.WriteImage(new_seg, os.path.join(new_seg_path, new_seg_name))

    # Prints the time that has been used once for each data processed
    print('already use {:.3f} min'.format((time() - start_time) / 60))
    print('-----------')

    file_index += 1
