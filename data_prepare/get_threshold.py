"""
Obtain a reasonable threshold to truncate the grayscale values of the original data to a certain range, thus reduce the influence of irrelevant data
"""

import os
import SimpleITK as sitk

ct_path = './dataset/AAATData/train/ct'
seg_path = './dataset/AAATData/train/label'

# Setting Thresholds
upper = 1200
lower = -300

num_point = 0.0  # Number of voxels belonging to aneurysm
num_inlier = 0.0  # Number of voxels in aneurysm voxels with gray value within the threshold value

for ct_file in os.listdir(ct_path):

    # Reading raw image and mask into memory
    ct = sitk.ReadImage(os.path.join(ct_path, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(seg_path, ct_file.replace('volume', 'segmentation')), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # Selection of voxels belonging to aneurysms
    organ_roi = ct_array[seg_array > 0]

    inliers = ((organ_roi <= upper) * (organ_roi >= lower)).astype(int).sum()

    print('{:.4}%'.format(inliers / organ_roi.shape[0] * 100))
    print('------------')

    num_point += organ_roi.shape[0]
    num_inlier += inliers

print(num_inlier / num_point)

# Maximum and minimum thresholds (1200, -300) for AAAT images
# Training set: 0.9987786950470848
# Testing set: 0.999382846239862
