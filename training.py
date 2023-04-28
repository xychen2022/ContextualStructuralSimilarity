#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
from trainer import Trainer

def get_arguments():
    
    parser = argparse.ArgumentParser(description="Segmentation via Contrastive Learning")
    parser.add_argument("--train-img-path", type=str, default="/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/images_uniform_v2/",
                        help="Path for training images.")
    parser.add_argument("--train-label-path", type=str, default="/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/label_uniform_v2/",
                        help="Path for training labels.")
    parser.add_argument("--test-img-path", type=str, default="/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/images_uniform_v2/",
                        help="Path for val/test images.")
    parser.add_argument("--test-label-path", type=str, default="/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/label_uniform_v2/",
                        help="Path for val/test labels.")
    parser.add_argument("--batch-size-train", type=int, default=2,
                        help="Number of image patches in each training step.")
    parser.add_argument("--batch-size-test", type=int, default=12,
                        help="Number of image patches in each testing step.")
    parser.add_argument("--patch-size", nargs='+', default=[80, 80, 80],
                        help="Size of training volume.")
    parser.add_argument("--num-of-downpooling", type=int, default=4,
                        help="Number of pooling operations (depends on specific architecture).")
    parser.add_argument("--num-input-channel", type=int, default=1,
                        help="Input channels.")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of epochs in the training process.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Starting epoch of training.")
    parser.add_argument("--num-samples-per-image", type=int, default=200,
                        help="Number of samples per image.")
    parser.add_argument("--num-val-samples-per-image", type=int, default=100,
                        help="Number of samples per image.")
    parser.add_argument("--lambda-ce", type=float, default=1,
                        help="Weight for cross entropy.")
    parser.add_argument("--lambda-csl", type=float, default=0.01,
                        help="Weight for contextual similarity loss.")
    parser.add_argument("--lambda-ssl", type=float, default=0.5,
                        help="Weight for structural similarity loss.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-train-images", type=int, default=62,
                        help="Number of images for training.")
    parser.add_argument("--list-train-image-ids", nargs='+', default=list(range(21, 41)) + list(range(41, 62)) + list(range(62, 83)),
                        help="Number of images for training.")
    parser.add_argument("--num-test-images", type=int, default=20,
                        help="Number of images for testing.")
    parser.add_argument("--list-test-image-ids", nargs='+', default=list(range(1, 21)),
                        help="Number of images for testing.")
    return parser.parse_args()
       
if __name__ == '__main__':
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    
    args = get_arguments()
    
    Trainer(args)