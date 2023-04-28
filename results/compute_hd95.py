#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:09:25 2020

@author: xychen
"""
from metrics import compute_surface_distances, compute_robust_hausdorff
from skimage.measure import label   

def getLargestCC(segmentation):
    labels = label(segmentation)

    #largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    largestCC = np.zeros_like(labels, dtype=np.float32)
    for idx in np.where(np.bincount(labels.flat)[1:] > 0.1*np.max(np.bincount(labels.flat)[1:]))[0]:
        largestCC[np.where(labels==(idx+1))] = 1
    
    return largestCC

def hd95(result, reference, voxelspacing=None):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    surface_distances = compute_surface_distances(reference, result, spacing_mm=voxelspacing)
    hd_dist_95 = compute_robust_hausdorff(surface_distances, 95)

    return hd_dist_95


import SimpleITK as sitk
import numpy as np

subject_list = list(range(1, 21))

pancreas = []
for idx in range(len(subject_list)):
    subject_id = subject_list[idx]
    print('Subject {0}'.format(subject_id))

    groundtruth = sitk.ReadImage('/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/label_uniform_v2/label{:04d}.nii.gz'.format(subject_id))
    spacing = groundtruth.GetSpacing()[::-1]
    print('spacing: ', spacing)
    groundtruth = sitk.GetArrayFromImage(groundtruth)

    groundtruth1 = np.zeros_like(groundtruth, dtype=np.float32)
    groundtruth1[np.where(groundtruth == 1)] = 1.
    groundtruth1 = getLargestCC(groundtruth1)
    groundtruth1 = groundtruth1.astype(np.bool)

    image = sitk.ReadImage('./segmentation/subject{0}/pred{0:04d}.nii.gz'.format(subject_id))
    image = sitk.GetArrayFromImage(image)

    label1 = np.zeros_like(image, dtype=np.float32)
    label1[np.where(image == 1)] = 1.
    label1 = getLargestCC(label1)
    label1 = label1.astype(np.bool)

    hausforffDis95 = hd95(label1, groundtruth1, voxelspacing= spacing)
    print('HD | mandible is: {}'.format(hausforffDis95))
    pancreas.append(hausforffDis95)

    print('\n')

print('pancreas: ', pancreas)
print(np.mean(pancreas), np.std(pancreas))


