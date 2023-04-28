#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
from cropping import corrected_crop
from network import VnetVariant
from skimage.measure import label

def getLargestCC(segmentation, threshold):
    labels = label(segmentation)

    largestCC = np.zeros_like(labels, dtype=np.float32)
    for idx in np.where(np.bincount(labels.flat)[1:] > threshold*np.max(np.bincount(labels.flat)[1:]))[0]:
        largestCC[np.where(labels==(idx+1))] = 1
    
    return largestCC

def Test(args):
    
    """
    To save time, a larger stride is used to get a coarse segmentation first;
    Then, a ROI is cropped and a smaller stride is used to get a final mask.
    """
    
    for j in range(len(args.list_test_image_ids)):
        subject_index = args.list_test_image_ids[j]
        
        print("Testing on subject {0}".format(subject_index))
        
        patch_size = np.array(args.patch_size) #2 * np.array(args.patch_size)
        num_of_downpooling = 4
        patch_stride_regulator = np.array([1, 1, 1])
        assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
        stride = patch_size//patch_stride_regulator
    
        model = VnetVariant(num_input_channel=args.num_input_channel, patch_size=args.patch_size, numofclasses=args.num_classes)
        model.load_weights(args.checkpoint_path)
        
        image = sitk.ReadImage(args.test_img_path + 'image{:04d}.nii.gz'.format(subject_index))
        image = sitk.GetArrayFromImage(image)
        image = image / np.max(image)
        
        original_size = image_size = np.array(np.shape(image))
        expanded_image_size = (np.ceil(image_size/(1.0*stride))*stride).astype(np.int32)
        
        expanded_image = np.zeros(expanded_image_size, dtype=np.float32)
        expanded_image[0:image_size[0], 0:image_size[1], 0:image_size[2]] = image
        
        predicted_seg = np.zeros([args.num_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
        count_matrix_seg = np.zeros([args.num_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
        num_of_patch_with_overlapping = (expanded_image_size/stride - patch_stride_regulator + 1).astype(np.int16)
        
        total_num_of_patches = np.prod(num_of_patch_with_overlapping)
        
        num_patch_z = num_of_patch_with_overlapping[0] # used for get patches
        num_patch_y = num_of_patch_with_overlapping[1] # used for get patches
        num_patch_x = num_of_patch_with_overlapping[2] # used for get patches
        
        print("Phase 1: total number of patches in the image is {0}".format(total_num_of_patches))
    
        center = np.zeros([total_num_of_patches, 3]) ## in the order of (total_num_of_patches, 3) ## (384, 3) ##
        
        patch_index = 0
        for ii in range(0, num_patch_z):
            for jj in range(0, num_patch_y):
                for kk in range(0, num_patch_x):
                    center[patch_index] = np.array([int(ii*stride[0] + patch_size[0]//2),
                                                    int(jj*stride[1] + patch_size[1]//2),
                                                    int(kk*stride[2] + patch_size[2]//2)])
                    patch_index += 1

        for idx in range(total_num_of_patches):
            
            image_one = np.zeros([patch_size[0], patch_size[1], patch_size[2]], dtype=np.float32)
            
            z_lower_bound  = int(center[idx][0] - patch_size[0]//2)
            z_higher_bound = int(center[idx][0] + patch_size[0]//2)
            y_lower_bound  = int(center[idx][1] - patch_size[1]//2)
            y_higher_bound = int(center[idx][1] + patch_size[1]//2)
            x_lower_bound  = int(center[idx][2] - patch_size[2]//2)
            x_higher_bound = int(center[idx][2] + patch_size[2]//2)
            
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
            
            image_one = np.expand_dims(np.expand_dims(image_one, axis=0), axis=0)
            
            ## output batch ##
            predicted_one, _ = model(image_one)
            
            predicted_seg[:, 
                          np.int16(center[idx][0] - patch_size[0]//2):np.int16(center[idx][0] + patch_size[0]//2),
                          np.int16(center[idx][1] - patch_size[1]//2):np.int16(center[idx][1] + patch_size[1]//2),
                          np.int16(center[idx][2] - patch_size[2]//2):np.int16(center[idx][2] + patch_size[2]//2)] += predicted_one[0]

            count_matrix_seg[:, 
                             np.int16(center[idx][0] - patch_size[0]//2):np.int16(center[idx][0] + patch_size[0]//2),
                             np.int16(center[idx][1] - patch_size[1]//2):np.int16(center[idx][1] + patch_size[1]//2),
                             np.int16(center[idx][2] - patch_size[2]//2):np.int16(center[idx][2] + patch_size[2]//2)] += 1.0

        predicted_seg_ = predicted_seg/(1.0*count_matrix_seg)
        
        output_seg = predicted_seg_[:, 0:image_size[0], 0:image_size[1], 0:image_size[2]]
        
        output_label = np.argmax(output_seg, axis=0)
        output_image_to_save = sitk.GetImageFromArray(output_label.astype(np.float32))

        if not os.path.exists(args.results_path + '/segmentation/subject{0}'.format(subject_index)):
            os.makedirs(args.results_path + '/segmentation/subject{0}'.format(subject_index))

        sitk.WriteImage(output_image_to_save, args.results_path + '/segmentation/' + 'subject{0}/coarse_pred{0:04d}.nii.gz'.format(subject_index))

        ## Second stage ##
        
        # Get the bounding box of ROI
        output_label    = getLargestCC(output_label, 0.2)
        depth_indicies  = np.where(output_label > 0)[0]
        height_indicies = np.where(output_label > 0)[1]
        width_indicies  = np.where(output_label > 0)[2]
        
        z1, z2 = np.min(depth_indicies), np.max(depth_indicies)
        y1, y2 = np.min(height_indicies), np.max(height_indicies)
        x1, x2 = np.min(width_indicies), np.max(width_indicies)
        
        crop_z_lower_bound  = max(0, z1-patch_size[0])
        crop_z_higher_bound = min(z2+patch_size[0], image_size[0])
        
        crop_y_lower_bound  = max(0, y1-patch_size[1])
        crop_y_higher_bound = min(y2+patch_size[1], image_size[1])
        
        crop_x_lower_bound  = max(0, x1-patch_size[2])
        crop_x_higher_bound = min(x2+patch_size[2], image_size[2])
        
        patch_size = np.array(args.patch_size)
        patch_stride_regulator = np.array([4, 4, 4]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
        assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
        stride = patch_size//patch_stride_regulator
        
        model = VnetVariant(num_input_channel=args.num_input_channel, patch_size=args.patch_size, numofclasses=args.num_classes)
        model.load_weights(args.checkpoint_path)
        
        image_crop = image[crop_z_lower_bound:crop_z_higher_bound, crop_y_lower_bound:crop_y_higher_bound, crop_x_lower_bound:crop_x_higher_bound]
        image_crop_size = np.array(np.shape(image_crop))
        print("image_crop_size = ", image_crop_size)
        
        expanded_image_size = np.maximum(patch_size, np.ceil(image_crop_size/(1.0*stride))*stride).astype(np.int32)
        predicted_seg = np.zeros([args.num_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
        count_matrix_seg = np.zeros([args.num_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)

        num_of_patch_with_overlapping = (expanded_image_size/stride - patch_stride_regulator + 1).astype(np.int16)
        
        total_num_of_patches = np.prod(num_of_patch_with_overlapping)
        
        num_patch_z = num_of_patch_with_overlapping[0] # used for get patches
        num_patch_y = num_of_patch_with_overlapping[1] # used for get patches
        num_patch_x = num_of_patch_with_overlapping[2] # used for get patches
        
        print("Phase 2: total number of patches in the image is {0}".format(total_num_of_patches))
        
        crop_z_lower_bound = crop_z_lower_bound - (expanded_image_size[0] - image_crop_size[0]) // 2
        crop_z_lower_bound = max(0, crop_z_lower_bound)
        crop_z_higher_bound = crop_z_lower_bound + expanded_image_size[0]
        crop_z_higher_bound = min(crop_z_higher_bound, image_size[0])
        
        crop_y_lower_bound = crop_y_lower_bound - (expanded_image_size[1] - image_crop_size[1]) // 2
        crop_y_lower_bound = max(0, crop_y_lower_bound)
        crop_y_higher_bound = crop_y_lower_bound + expanded_image_size[1]
        crop_y_higher_bound = min(crop_y_higher_bound, image_size[1])
        
        crop_x_lower_bound = crop_x_lower_bound - (expanded_image_size[2] - image_crop_size[2]) // 2
        crop_x_lower_bound = max(0, crop_x_lower_bound)
        crop_x_higher_bound = crop_x_lower_bound + expanded_image_size[2]
        crop_x_higher_bound = min(crop_x_higher_bound, image_size[2])
        
        expanded_image = np.zeros(expanded_image_size, dtype=np.float32)
        expanded_image[:(crop_z_higher_bound-crop_z_lower_bound),
                       :(crop_y_higher_bound-crop_y_lower_bound),
                       :(crop_x_higher_bound-crop_x_lower_bound)] = image[crop_z_lower_bound:crop_z_higher_bound,
                                                                          crop_y_lower_bound:crop_y_higher_bound,
                                                                          crop_x_lower_bound:crop_x_higher_bound]
    
        center = np.zeros([total_num_of_patches, 3])
        
        patch_index = 0
        for ii in range(0, num_patch_z):
            for jj in range(0, num_patch_y):
                for kk in range(0, num_patch_x):
                    center[patch_index] = np.array([int(ii*stride[0] + patch_size[0]//2),
                                                    int(jj*stride[1] + patch_size[1]//2),
                                                    int(kk*stride[2] + patch_size[2]//2)])
                    patch_index += 1
        
        modulo=np.mod(total_num_of_patches, args.batch_size_test)
        if modulo!=0:
            num_to_add=args.batch_size_test-modulo
            inds_to_add=np.random.randint(0, total_num_of_patches, num_to_add) ## the return value is a ndarray
            to_add = center[inds_to_add]
            new_center = np.vstack((center, to_add))
        else:
            new_center = center
        
        np.random.shuffle(new_center)
        for i_batch in range(int(new_center.shape[0]/args.batch_size_test)):
            label = 0
            subvertex = new_center[i_batch*args.batch_size_test:(i_batch+1)*args.batch_size_test]
            for count in range(args.batch_size_test):
                
                image_one = np.zeros([patch_size[0], patch_size[1], patch_size[2]], dtype=np.float32)
                
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
            
            ## output batch ##
            predicted_one, _ = model(Img_1)
            
            for idx in range(args.batch_size_test):
                
                predicted_seg[:, np.int16(subvertex[idx][0] - patch_size[0]//2):np.int16(subvertex[idx][0] + patch_size[0]//2),
                                np.int16(subvertex[idx][1] - patch_size[1]//2):np.int16(subvertex[idx][1] + patch_size[1]//2),
                                np.int16(subvertex[idx][2] - patch_size[2]//2):np.int16(subvertex[idx][2] + patch_size[2]//2)] += predicted_one[idx]
    
                count_matrix_seg[:, np.int16(subvertex[idx][0] - patch_size[0]//2):np.int16(subvertex[idx][0] + patch_size[0]//2),
                                     np.int16(subvertex[idx][1] - patch_size[1]//2):np.int16(subvertex[idx][1] + patch_size[1]//2),
                                     np.int16(subvertex[idx][2] - patch_size[2]//2):np.int16(subvertex[idx][2] + patch_size[2]//2)] += 1.0

        predicted_seg_ = predicted_seg/(1.0*count_matrix_seg)
        
        output_seg = predicted_seg_[:, 
                                    0:(crop_z_higher_bound-crop_z_lower_bound), 
                                    0:(crop_y_higher_bound-crop_y_lower_bound), 
                                    0:(crop_x_higher_bound-crop_x_lower_bound)]
        output_label = np.argmax(output_seg, axis=0)
        
        map_to_original = np.zeros(list(original_size), dtype=np.float32)
        map_to_original[crop_z_lower_bound:crop_z_higher_bound,
                        crop_y_lower_bound:crop_y_higher_bound,
                        crop_x_lower_bound:crop_x_higher_bound] = output_label
        
        output_image_to_save = sitk.GetImageFromArray(map_to_original.astype(np.float32))

        if not os.path.exists(args.results_path + '/segmentation/subject{0}'.format(subject_index)):
            os.makedirs(args.results_path + '/segmentation/subject{0}'.format(subject_index))

        sitk.WriteImage(output_image_to_save, args.results_path + '/segmentation/' + 'subject{0}/pred{0:04d}.nii.gz'.format(subject_index))
