#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import math
import random
import numpy as np
import SimpleITK as sitk

def affine_transform(itk_image, itk_label):
    
    imageFilter = sitk.MinimumMaximumImageFilter()
    imageFilter.Execute(itk_image)
    
    #original_size = itk_image.GetSize()
    width, height, depth = itk_image.GetSize()
    original_spacing = itk_image.GetSpacing()
    
    rotation_center = itk_image.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                                               int(np.ceil(height/2)),
                                                               int(np.ceil(depth/2))))
    theta_z = random.uniform(-math.pi/6.0, math.pi/6.0)
    theta_y = random.uniform(-math.pi/6.0, math.pi/6.0)
    theta_x = random.uniform(-math.pi/6.0, math.pi/6.0)
    translation = [random.randint(-25, 25) for i in range(3)]
    scale_factor = random.uniform(0.8, 1.2)
    similarity = sitk.Euler3DTransform(rotation_center, theta_x, theta_y, theta_z, translation)
    
    T=sitk.AffineTransform(3)
    T.SetMatrix(similarity.GetMatrix())
    T.SetCenter(similarity.GetCenter())
    T.SetTranslation(similarity.GetTranslation())
    T.Scale(scale_factor)
    
    newSpacing = np.array(original_spacing) / scale_factor
    
    resampler1 = sitk.ResampleImageFilter()
    resampler1.SetReferenceImage(itk_image)
    resampler1.SetInterpolator(sitk.sitkLinear)
    resampler1.SetDefaultPixelValue(imageFilter.GetMinimum())
    resampler1.SetTransform(T)
    
    resampler2 = sitk.ResampleImageFilter()
    resampler2.SetReferenceImage(itk_image)
    resampler2.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler2.SetDefaultPixelValue(0)
    resampler2.SetTransform(T)
 
    imgResampled = resampler1.Execute(itk_image)
    
    imgResampled.SetSpacing(newSpacing)
    imgResampled.SetDirection(itk_image.GetDirection())
    imgResampled.SetOrigin(itk_image.GetOrigin())
    
    labResampled = resampler2.Execute(itk_label)
    
    labResampled.SetSpacing(newSpacing)
    labResampled.SetDirection(itk_image.GetDirection())
    labResampled.SetOrigin(itk_image.GetOrigin())
    
    return imgResampled, labResampled

