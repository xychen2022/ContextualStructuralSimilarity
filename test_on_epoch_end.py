#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import shutil
import numpy as np
import SimpleITK as sitk
from cropping import corrected_crop

def test(model, img_path, label_path, subject_list, patch_size, batch_size=1, numClasses=2, epoch=0, current_best=0.):
    # the following parameters need to be assigned values before training
    patch_size = np.array(patch_size) # very important
    num_of_downpooling = 4 # very important
    patch_stride_regulator = np.array([2, 2, 2]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    stride = patch_size/patch_stride_regulator
    
    tmp_path = os.path.join(os.getcwd(), 'tmp')
    best_path = os.path.join(os.getcwd(), 'best_results')
    
    dice_coef_list = []
    for j in range(len(subject_list)):
        subject_index = subject_list[j]
                
        ### compute basic parameter for usage later
        image = sitk.ReadImage(img_path + 'image{:04d}.nii.gz'.format(subject_index))
        image = sitk.GetArrayFromImage(image)
        image = ( image - np.min(image) ) / ( np.max(image) - np.min(image) )
        
        image_size = np.array(np.shape(image))
                
        expanded_image_size = (np.ceil(image_size/(1.0*stride))*stride).astype(np.int32)
                
        expanded_image = np.zeros(expanded_image_size, dtype=np.float32)
        expanded_image[0:image_size[0], 0:image_size[1], 0:image_size[2]] = image
        
        predicted_seg = np.zeros([numClasses, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)

        count_matrix_seg = np.zeros([numClasses, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
        
        num_of_patch_with_overlapping = expanded_image_size/stride - patch_stride_regulator + 1
        
        total_num_of_patches = np.prod(num_of_patch_with_overlapping)
        
        num_patch_z = int(num_of_patch_with_overlapping[0]) # used for get patches
        num_patch_y = int(num_of_patch_with_overlapping[1]) # used for get patches
        num_patch_x = int(num_of_patch_with_overlapping[2]) # used for get patches
        
        center = np.zeros([total_num_of_patches.astype(np.int32), 3])
        
        patch_index = 0
        for ii in range(0, num_patch_z):
            for jj in range(0, num_patch_y):
                for kk in range(0, num_patch_x):
                    center[patch_index] = np.array([int(ii*stride[0] + patch_size[0]//2),
                                                    int(jj*stride[1] + patch_size[1]//2),
                                                    int(kk*stride[2] + patch_size[2]//2)])
                    patch_index += 1
        
        modulo=np.mod(total_num_of_patches.astype(np.int32), batch_size)
        if modulo!=0:
            num_to_add=batch_size-modulo
            inds_to_add=np.random.randint(0, int(total_num_of_patches.astype(np.int32)), int(num_to_add)) ## the return value is a ndarray
            to_add = center[inds_to_add]
            new_center = np.vstack((center, to_add))
        else:
            new_center = center
        
        np.random.shuffle(new_center)
        for i_batch in range(int(new_center.shape[0]/batch_size)):
            label = 0
            subvertex = new_center[i_batch*batch_size:(i_batch+1)*batch_size]
            for count in range(batch_size):
                ## 96*96*96 ##
                image_one = np.zeros([int(patch_size[0]), int(patch_size[1]), int(patch_size[2])], dtype=np.float32)
                
                z_lower_bound = int(subvertex[count][0] - patch_size[0]//2)
                z_higher_bound = int(subvertex[count][0] + patch_size[0]//2)
                y_lower_bound = int(subvertex[count][1] - patch_size[1]//2)
                y_higher_bound = int(subvertex[count][1] + patch_size[1]//2)
                x_lower_bound = int(subvertex[count][2] - patch_size[2]//2)
                x_higher_bound = int(subvertex[count][2] + patch_size[2]//2)
                
                virgin_range = np.array([z_lower_bound, z_higher_bound, y_lower_bound, y_higher_bound, x_lower_bound, x_higher_bound])
                copy_from, copy_to = corrected_crop(virgin_range, expanded_image_size)
                
                cf_z_lower_bound = int(copy_from[0])
                if copy_from[1] is not None:
                    cf_z_higher_bound = int(copy_from[1])
                else:
                    cf_z_higher_bound = None
                
                cf_y_lower_bound = int(copy_from[2])
                if copy_from[3] is not None:
                    cf_y_higher_bound = int(copy_from[3])
                else:
                    cf_y_higher_bound = None
                
                cf_x_lower_bound = int(copy_from[4])
                if copy_from[5] is not None:
                    cf_x_higher_bound = int(copy_from[5])
                else:
                    cf_x_higher_bound = None
                
                image_one[int(copy_to[0]):copy_to[1],
                          int(copy_to[2]):copy_to[3],
                          int(copy_to[4]):copy_to[5]] = \
                          expanded_image[cf_z_lower_bound:cf_z_higher_bound,
                                         cf_y_lower_bound:cf_y_higher_bound,
                                         cf_x_lower_bound:cf_x_higher_bound]
            
                image_one = np.expand_dims(image_one, axis=0)
                image_1 = np.expand_dims(image_one, axis=0)
                
                if label == 0:
                    Img_1 = image_1
                    label += 1
                else:
                    Img_1 = np.vstack((Img_1, image_1))
                    label += 1
            
            predicted_one, _ = model(Img_1)
                  
            for idx in range(batch_size):
                
                predicted_seg[:, 
                              int(subvertex[idx][0] - patch_size[0]//2):int(subvertex[idx][0] + patch_size[0]//2),
                              int(subvertex[idx][1] - patch_size[1]//2):int(subvertex[idx][1] + patch_size[1]//2),
                              int(subvertex[idx][2] - patch_size[2]//2):int(subvertex[idx][2] + patch_size[2]//2)] += predicted_one[idx]
    
                count_matrix_seg[:, 
                                 int(subvertex[idx][0] - patch_size[0]//2):int(subvertex[idx][0] + patch_size[0]//2),
                                 int(subvertex[idx][1] - patch_size[1]//2):int(subvertex[idx][1] + patch_size[1]//2),
                                 int(subvertex[idx][2] - patch_size[2]//2):int(subvertex[idx][2] + patch_size[2]//2)] += 1.0
        
        predicted_seg_ = predicted_seg/(1.0*count_matrix_seg)
        
        output_seg = predicted_seg_[:, 0:image_size[0], 0:image_size[1], 0:image_size[2]]

        output_label = np.argmax(output_seg, axis=0)
        
        label1 = np.zeros_like(image, dtype=np.float32)
        label1[np.where(output_label == 1)] = 1.
        
        groundtruth = sitk.ReadImage(label_path + 'label{:04d}.nii.gz'.format(subject_index))
        groundtruth = sitk.GetArrayFromImage(groundtruth)
        
        groundtruth1 = np.zeros_like(groundtruth, dtype=np.float32)
        groundtruth1[np.where(groundtruth == 1)] = 1.
    
        intersection = np.logical_and(label1, groundtruth1)
        im_sum = label1.sum() + groundtruth1.sum()
    
        dice_coef_list.append(2. * intersection.sum() / im_sum)
                
        output_image_to_save = sitk.GetImageFromArray(output_label.astype(np.float32))

        if not os.path.exists(os.path.join(tmp_path, 'subject{0}'.format(subject_index))):
            os.makedirs(os.path.join(tmp_path, 'subject{0}'.format(subject_index)))

        sitk.WriteImage(output_image_to_save, os.path.join(tmp_path, 'subject{0}/pred{0:04d}.nii.gz'.format(subject_index)))
    
    print('dice_coef_list: ', dice_coef_list, '\nmean: ', np.mean(dice_coef_list), ' std: ', np.std(dice_coef_list))
    current_mean = np.mean(dice_coef_list)
    
    if current_mean > current_best:
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        
        shutil.copytree(tmp_path, best_path)
        
        model.save_weights('./checkpoints/supervised_model_best.ckpt')
        model.save_weights('./checkpoints/supervised_model_epoch{0}_avg{1:5f}.ckpt'.format(epoch+1, current_mean))
        return current_mean
    else:
        return current_best