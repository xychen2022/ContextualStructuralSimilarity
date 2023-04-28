#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
from predict_on_given_set import Test

def get_arguments():
    
    parser = argparse.ArgumentParser(description="Segmentation via Contrastive Learning")
    parser.add_argument("--test-img-path", type=str, default="/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/images_uniform_v2/",
                        help="Path for val/test images.")
    parser.add_argument("--test-label-path", type=str, default="/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/label_uniform_v2/",
                        help="Path for val/test labels.")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/supervised_model_best.ckpt",
                        help="Number of image patches in each testing step.")
    parser.add_argument("--results-path", type=str, default="./results",
                        help="Number of image patches in each testing step.")
    parser.add_argument("--batch-size-test", type=int, default=12,
                        help="Number of image patches in each testing step.")
    parser.add_argument("--patch-size", nargs='+', default=[80, 80, 80],
                        help="Size of training volume.")
    parser.add_argument("--num-input-channel", type=int, default=1,
                        help="Input channels.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-test-images", type=int, default=20,
                        help="Number of images for testing.")
    parser.add_argument("--list-test-image-ids", nargs='+', default=list(range(1, 21)),
                        help="Number of images for testing.")
    return parser.parse_args()
       
if __name__ == '__main__':
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    
    args = get_arguments()
    
    Test(args)