"""
Test Script
"""

import os
from time import time
from time import sleep
from tqdm import tqdm

import torch
import torch.nn.functional as F
import copy

import numpy as np
import SimpleITK as sitk
import xlsxwriter as xw
import scipy.ndimage as ndimage

from net.CACUNet import Net

import skimage.measure as measure
import skimage.morphology as morphology
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import (closing, cube, binary_closing)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

test_ct_dir = './dataset/AAATData/test/ct'
test_seg_dir = './dataset/AAATData/test/label'

aneurysm_pred_dir = './pred/'

module_dir = './model/Bestnet467-0.105-0.148.pth'

upper = 1200
lower = -300
down_scale = 0.5
size = 48
slice_thickness = 1
expand_pixel = 0
overlap = 6 #6
overlap_pixel = 8 #8
Cut_process = True # True or False
CCO_process = False # True or False
maximum_hole = 5e4

aneurysm_list = [
    'Lumen',
    'Thrombosis',
]

# Write evaluation indicators to excel file
workbook = xw.Workbook('./result.xlsx')
worksheet = workbook.add_worksheet('result')

# Setting the cell formatting
bold = workbook.add_format()
bold.set_bold()

center = workbook.add_format()
center.set_align('center')

center_bold = workbook.add_format()
center_bold.set_bold()
center_bold.set_align('center')

worksheet.set_column(1, len(os.listdir(test_ct_dir)), width=4)
worksheet.set_column(0, 0, width=30, cell_format=center_bold)
worksheet.set_row(0, 20, center_bold)

# Write file name
worksheet.write(0, 0, 'file name')
for index, file_name in enumerate(os.listdir(test_ct_dir), start=1):
    worksheet.write(0, index, file_name)

# Write the name of each evaluation indicator
for index, aneurysm_name in enumerate(aneurysm_list, start=1):
    worksheet.write(index, 0, aneurysm_name)
worksheet.write(3, 0, 'speed')
worksheet.write(4, 0, 'shape')


# Define the network and load the parameters
net = torch.nn.DataParallel(Net(training=False)).cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()

avg_lumen = 0
avg_thrombosis = 0

def doCut(Binary_Img):
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
    labeled_Img = labeled_Img.astype(bool)
    morphology.remove_small_holes(labeled_Img, maximum_hole, connectivity=2, in_place=True)

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


# Starting tests
for file_index, file in enumerate(os.listdir(test_ct_dir)):

    start_time = time()
    
    # Read 3D images into memory
    ct = sitk.ReadImage(os.path.join(test_ct_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    overlap_pred_seg = np.zeros((ct_array.shape[0], ct_array.shape[1], ct_array.shape[2]), dtype=np.int8)
    overlap_seg = np.zeros((ct_array.shape[0], ct_array.shape[1], ct_array.shape[2]), dtype=np.int8)

    # Overlap
    pbar = tqdm(total=overlap,desc='Overlap')

    for overlap_index in range (0, overlap):
    
        # Block sampling in the axial direction
        pbar.update(1)
        sleep(0.1)
        flag = False
        start_slice = 0 + overlap_index*overlap_pixel
        end_slice = start_slice + size - 1 
        ct_array_list = []

        while end_slice < ct_array.shape[0] - 1:
            ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])

            start_slice = end_slice + 1
            end_slice = start_slice + size - 1
        
        if end_slice == ct_array.shape[0] - 1:
            ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])
        else:
            flag = True
            count = ct_array.shape[0] - start_slice
            ct_array_list.append(ct_array[-size:, :, :])

        if len(ct_array_list) == 1:
            print('The remaining size is less than 48, the overlap is break!')
            break

        outputs_list = []
        with torch.no_grad():
            for ct_block in ct_array_list:

                ct_tensor = torch.FloatTensor(ct_block).cuda()
                ct_tensor = ct_tensor.unsqueeze(dim=0)
                ct_tensor = ct_tensor.unsqueeze(dim=0)

                outputs = net(ct_tensor)
                outputs = outputs.squeeze()

                outputs_list.append(outputs.cpu().detach().numpy())
                del outputs

        # After execution, start stitching the results
        pred_seg = np.concatenate(outputs_list[0:-1], axis=1)
        if flag is False:
            pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=1)
        else:
            pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, -count:, :, :]], axis=1)
        
        pred_seg = np.argmax(pred_seg, axis=0)
        pred_seg = np.round(pred_seg).astype(np.uint8)
        overlap_pred_seg[0+overlap_index*overlap_pixel:overlap_pred_seg.shape[0],:,:] = pred_seg
        
        if overlap_index == 0:
            overlap_seg = overlap_pred_seg
        else:
            Overlap_Img = np.zeros(np.shape(overlap_seg))
            overlap_C1 = np.int8(overlap_seg == 1)
            overlap_C2 = np.int8(overlap_seg == 2)
            overlap_pred_C1 = np.int8(overlap_pred_seg == 1)
            overlap_pred_C2 = np.int8(overlap_pred_seg == 2)
            overlap_C1 = 2 * overlap_C1 - overlap_pred_C1
            overlap_C1[overlap_C1 == 2] = 1
            overlap_C1[overlap_C1 == -1] = 1
            overlap_C1 = overlap_C1.astype(bool)
            #Overlap_Img[overlap_C1] = 1

            overlap_C2 = 2 * overlap_C2 - overlap_pred_C2
            overlap_C2[overlap_C2 == 2] = 1
            overlap_C2[overlap_C2 == -1] = 1
            overlap_C2 = overlap_C2.astype(bool)
            Overlap_Img[overlap_C2] = 2
            Overlap_Img[overlap_C1] = 1

            overlap_seg = Overlap_Img

        del pred_seg
        del ct_array_list
    
    pbar.close()

    pred_seg = overlap_seg

    # Reading the gold standard into memory to calculate dice coefficients
    seg = sitk.ReadImage(os.path.join(test_seg_dir, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    
    mask_Seg = copy.deepcopy(seg_array)
    mask_Seg[mask_Seg > 0] = 1

    mask_Seg  =doCut(mask_Seg)

    #post-processing
    mask_Img = copy.deepcopy(mask_Seg)

    # Reduce post-processing complexity by cutting out areas of predicted results
    z = np.any(mask_Img, axis=(1, 2))
    start_z, end_z = np.where(z)[0][[0, -1]]

    y = np.any(mask_Img, axis=(0, 1))
    start_y, end_y = np.where(y)[0][[0, -1]]

    x = np.any(mask_Img, axis=(0, 2))
    start_x, end_x = np.where(x)[0][[0, -1]]
        
    # Extending slices
    start_z = max(0, start_z - expand_pixel)
    start_x = max(0, start_x - expand_pixel)
    start_y = max(0, start_y - expand_pixel)

    end_z = min(mask_Img.shape[0]-1, end_z + expand_pixel)
    end_x = min(mask_Img.shape[1]-1, end_x + expand_pixel)
    end_y = min(mask_Img.shape[2]-1, end_y + expand_pixel) 
    
    print(start_z, end_z+1, start_x, end_x+1, start_y, end_y+1)

    new_pred_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], pred_seg.shape[2]), dtype=np.float32)
    new_pred_seg[start_z: end_z+1, start_x: end_x+1, start_y: end_y+1] = 1

    # Cut
    if Cut_process is True:
        pred_seg = new_pred_seg*pred_seg
    pred_seg = pred_seg.astype(np.uint8)
    
    if Cut_process is True:
        seg_array = new_pred_seg*seg_array
    seg_array = seg_array.astype(np.uint8)

    # CCO
    if CCO_process is True:
        pred_seg = CCO(pred_seg)
        pred_seg = pred_seg.astype(np.uint8)
    
    print('size of pred: ', pred_seg.shape)
    print('size of GT: ', seg_array.shape)

    worksheet.write(4, file_index + 1, pred_seg.shape[0])

    # Calculate the dice coefficient for each type of aneurysm and write the results in a table
    for aneurysm_index, aneurysm in enumerate(aneurysm_list, start=1):

        pred_aneurysm = np.zeros(pred_seg.shape)
        target_aneurysm = np.zeros(seg_array.shape)

        pred_aneurysm[pred_seg == aneurysm_index] = 1
        target_aneurysm[seg_array == aneurysm_index] = 1

        # If there is no aneurysm of a certain type, record None in the table and skip it.
        if target_aneurysm.sum() == 0:
            worksheet.write(aneurysm_index, file_index + 1, 'None')

        else:
            dice = (2 * pred_aneurysm * target_aneurysm).sum() / (pred_aneurysm.sum() + target_aneurysm.sum())
            worksheet.write(aneurysm_index, file_index + 1, dice)

        if aneurysm_index == 1:
            avg_lumen = avg_lumen + dice
            print(aneurysm,'Dice:{:.4f}'.format(dice))
        
        if aneurysm_index == 2:
            avg_thrombosis = avg_thrombosis + dice
            print(aneurysm,'Dice:{:.4f}'.format(dice))

    # Save the predicted results as nii data
    pred_seg = sitk.GetImageFromArray(pred_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(aneurysm_pred_dir, file.replace('volume', 'pred')))
    del pred_seg

    speed = time() - start_time

    worksheet.write(3, file_index + 1, speed)

    print('this case use {:.3f} s'.format(speed))

    print('-----------------------')

workbook.close()
print('Avg_Dice_Lumen:{:.4f}'.format(avg_lumen/(file_index+1)), 'Avg_Dice_Thrombosis:{:.4f}'.format(avg_thrombosis/(file_index+1)))
