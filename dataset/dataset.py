"""
Random sampling of the dataset
"""

import os
import random

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset

on_server = True
size = 48


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('img', 'label'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

    def __getitem__(self, index):
        """
        :param index:
        :return: torch.Size([B, 1, 48, 512, 512]) torch.Size([B, 48, 512, 512])
        """

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # Read raw image and mask into memory
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # Randomly select 48 slices 
        start_slice = random.randint(0, ct_array.shape[0] - size)
        end_slice = start_slice + size - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        # Convert array to tensor after processing
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)


ct_dir = '/home/CACU-Net/train/CT/' \
    if on_server is False else './train/CT/'
seg_dir = '/home/CACU-Net/train/GT/' \
    if on_server is False else './train/GT/'

train_ds = Dataset(ct_dir, seg_dir)

