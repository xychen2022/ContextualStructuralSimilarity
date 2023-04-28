#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import SimpleITK as sitk
from cropping import corrected_crop
from transform import affine_transform
from tensorflow.keras.utils import to_categorical

def train_generator(img_path, label_path, subject_list, patch_size, num_of_downpooling, batch_size=1, numClasses=2, numImages=62, numSamples=200, numAugs=5):
    # the following parameters need to be assigned values before training
    patch_size = np.array(patch_size)
    subject_list = np.array(subject_list)
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    
    assert len(subject_list) == numImages
    label = 0
    
    while True:
        np.random.shuffle(subject_list)
        for j in range(len(subject_list)):
            
            subject_index = subject_list[j]
            image_pre_aug = sitk.ReadImage(img_path + 'image{:04d}.nii.gz'.format(subject_index))
            segmentation_gt_pre_aug = sitk.ReadImage(label_path + 'label{:04d}.nii.gz'.format(subject_index))
            
            assert numSamples % numAugs == 0 and numSamples // numAugs % batch_size == 0
            
            for aug_i in range(numAugs):
                # Augmentation #
                image, segmentation_gt = affine_transform(image_pre_aug, segmentation_gt_pre_aug)
                
                image = sitk.GetArrayFromImage(image)
                image = ( image - np.min(image) ) / ( np.max(image) - np.min(image) )
                
                segmentation_gt = sitk.GetArrayFromImage(segmentation_gt)
                
                unique_ids = np.unique(segmentation_gt)
                nonzero_unique_ids = unique_ids[np.where(unique_ids > 0)]
                assert np.all(np.array([x < numClasses for x in nonzero_unique_ids]))
                assert nonzero_unique_ids is not np.array([])
                
                segmentation_gt = to_categorical(segmentation_gt, numClasses).reshape(list(segmentation_gt.shape + (numClasses,)))
                segmentation_gt = np.transpose(segmentation_gt, [3, 0, 1, 2])
                
                image_size = image.shape
                seg_gt_size = segmentation_gt.shape
                assert seg_gt_size[1] == image_size[0] and seg_gt_size[2] == image_size[1] and seg_gt_size[3] == image_size[2]
                
                numSamplesPerAug = numSamples // numAugs
                vertex = np.zeros([numSamplesPerAug, 2, 6])
                shapedata = vertex.shape
                
                patch_index = 0 ## update by 1 after generating a patch
                margin = 10
                assert np.all(np.array([margin < size_//2 for size_ in patch_size]))
                
                for label_i in nonzero_unique_ids:
                    m = segmentation_gt[int(label_i)]
                    # Bounding box.
                    depth_indicies = np.where(m == 1)[0]
                    height_indicies = np.where(m == 1)[1]
                    width_indicies = np.where(m == 1)[2]
                    
                    z1, z2 = np.min(depth_indicies), np.max(depth_indicies)
                    y1, y2 = np.min(height_indicies), np.max(height_indicies)
                    x1, x2 = np.min(width_indicies), np.max(width_indicies)
                    
                    for ii in range(int(0.8*numSamplesPerAug/len(nonzero_unique_ids))):
                        center_z = np.random.randint(z1-margin, z2+margin+1)
                        center_y = np.random.randint(y1-margin, y2+margin+1)
                        center_x = np.random.randint(x1-margin, x2+margin+1)
                        
                        vertex[patch_index][0] = np.array([center_z-int(patch_size[0]//2), center_z+int(patch_size[0]//2),
                                                           center_y-int(patch_size[1]//2), center_y+int(patch_size[1]//2),
                                                           center_x-int(patch_size[2]//2), center_x+int(patch_size[2]//2)])
    
                        vertex[patch_index][1] = np.array([center_z-2*int(patch_size[0]//2), center_z+2*int(patch_size[0]//2),
                                                           center_y-2*int(patch_size[1]//2), center_y+2*int(patch_size[1]//2),
                                                           center_x-2*int(patch_size[2]//2), center_x+2*int(patch_size[2]//2)])
    
                        patch_index += 1
                            
                while patch_index < numSamplesPerAug:
    
                    center_z = np.random.randint(int(patch_size[0]//4), image_size[0]-int(patch_size[0]//4))
                    center_y = np.random.randint(int(patch_size[1]//4), image_size[1]-int(patch_size[1]//4))
                    center_x = np.random.randint(int(patch_size[2]//4), image_size[2]-int(patch_size[2]//4))
                    
                    vertex[patch_index][0] = np.array([center_z-int(patch_size[0]//2), center_z+int(patch_size[0]//2),
                                                       center_y-int(patch_size[1]//2), center_y+int(patch_size[1]//2),
                                                       center_x-int(patch_size[2]//2), center_x+int(patch_size[2]//2)])
                    
                    vertex[patch_index][1] = np.array([center_z-2*int(patch_size[0]//2), center_z+2*int(patch_size[0]//2),
                                                       center_y-2*int(patch_size[1]//2), center_y+2*int(patch_size[1]//2),
                                                       center_x-2*int(patch_size[2]//2), center_x+2*int(patch_size[2]//2)])
                    
                    patch_index += 1
                
                modulo=np.mod(shapedata[0], batch_size)
                if modulo!=0:
                    num_to_add=batch_size-modulo
                    inds_to_add=np.random.randint(0, shapedata[0], num_to_add)
                    to_add = vertex[inds_to_add]
                    new_vertex = np.vstack((vertex, to_add))
                else:
                    new_vertex = vertex
                
                np.random.shuffle(new_vertex)
                for i_batch in range(int(new_vertex.shape[0]/batch_size)):
                    subvertex = new_vertex[i_batch*batch_size:(i_batch+1)*batch_size]
                    for count in range(batch_size):
                        ## size_*size_*size_ ##
                        image_one = np.zeros([int(patch_size[0]), int(patch_size[1]), int(patch_size[2])], dtype=np.float32)
                        seg_gt_one = np.zeros([numClasses, int(patch_size[0]), int(patch_size[1]), int(patch_size[2])], dtype=np.float32)
                        seg_gt_one[0] = np.ones([int(patch_size[0]), int(patch_size[1]), int(patch_size[2])], dtype=np.float32) ## I made a huge mistake here ##
                        
                        copy_from, copy_to = corrected_crop(subvertex[count][0], np.array(list(image_size)))
    
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
                                  image[cf_z_lower_bound:cf_z_higher_bound,
                                        cf_y_lower_bound:cf_y_higher_bound,
                                        cf_x_lower_bound:cf_x_higher_bound]
    
                        seg_gt_one[:,
                                   int(copy_to[0]):copy_to[1],
                                   int(copy_to[2]):copy_to[3],
                                   int(copy_to[4]):copy_to[5]] = \
                                   segmentation_gt[:,
                                                    cf_z_lower_bound:cf_z_higher_bound,
                                                    cf_y_lower_bound:cf_y_higher_bound,
                                                    cf_x_lower_bound:cf_x_higher_bound]
    
                        image_one = np.expand_dims(image_one, axis=0)
                        
                        gtClass = np.argmax(seg_gt_one, axis=0)
                        
                        blockSize = 2**num_of_downpooling
                        numBlocks = (patch_size // blockSize).astype(np.int32)
                        
                        classBlocks = []
                        for idx_z in range(int(numBlocks[0])):
                            for idx_y in range(int(numBlocks[1])):
                                for idx_x in range(int(numBlocks[2])):
                                    block = gtClass[idx_z*blockSize:(idx_z+1)*blockSize, idx_y*blockSize:(idx_y+1)*blockSize, idx_x*blockSize:(idx_x+1)*blockSize]
                                    uniqueBlock = list(np.unique(block))
                                    
                                    classBlock = np.zeros([numClasses,], dtype=np.float32)
                                    for idx in uniqueBlock:
                                        classBlock[idx] = 1
                                    
                                    classBlocks.append(classBlock)
                        classBlocks = np.array(classBlocks)
                        
                        classBlocks = np.expand_dims(classBlocks, axis=0)
                        
                        ## output batch ##
                        image_1 = np.expand_dims(image_one, axis=0)
                        seg_gt_one = np.expand_dims(seg_gt_one, axis=0)
    
                        if label == 0:
                            Img_1 = image_1
                            seg_gt = seg_gt_one
                            class_gt = classBlocks
                            label += 1
                        else:
                            Img_1 = np.vstack((Img_1, image_1))
                            seg_gt = np.vstack((seg_gt, seg_gt_one))
                            class_gt = np.vstack((class_gt, classBlocks))
                            label += 1
                        
                        if np.remainder(label, batch_size)==0:
                            yield [Img_1, seg_gt, class_gt]
                            label = 0
