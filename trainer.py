#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from dataloader import train_generator
from test_on_epoch_end import test

from network import VnetVariant
from losses import custom_categorical_crossentropy, CSL, SSL

def learning_rate_scheduler(opt, epoch, total_epochs):
    if epoch > total_epochs // 3 * 2:
        opt.lr = 1e-5
    elif epoch > total_epochs // 3:
        opt.lr = 5e-5
    else:
        opt.lr = 1e-4

def Trainer(args):
    
    iterations = args.num_samples_per_image * args.num_train_images // args.batch_size_train
    
    model = VnetVariant(num_input_channel=args.num_input_channel, patch_size=args.patch_size, numofclasses=args.num_classes)
    
    train_gen = train_generator(img_path=args.train_img_path, label_path=args.train_label_path, subject_list=args.list_train_image_ids, patch_size=args.patch_size, num_of_downpooling=args.num_of_downpooling, batch_size=args.batch_size_train, numClasses=args.num_classes, numImages=args.num_train_images, numSamples=args.num_samples_per_image)
    
    #optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-4, global_clipnorm=5.0)
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    
    current_best_mean_dice = 0
    
    # start from checkpoints
    if args.start_epoch > 0:
        model.load_weights('./checkpoints/supervised_model_best.ckpt')
    
    for epoch in range(args.start_epoch, args.num_epochs):
        
        #learning_rate_scheduler(optimizer, epoch, args.num_epochs)
        
        losses_ce = []
        losses_csl = []
        losses_ssl = []
        
        for i_iter in range(iterations):
            
            # Train with labeled data
            image1, batch_label, batch_gtClass = next(train_gen)
            
            with tf.GradientTape() as tape:
                probs, local_context = model(image1)
                
                loss_ce = custom_categorical_crossentropy(batch_label, probs)
                loss_csl = CSL(batch_gtClass, local_context, alpha=0.25, gamma=2.0)
                loss_ssl = SSL(batch_label, probs, num_classes=args.num_classes, margin=2)
                
                loss = args.lambda_ce * loss_ce + args.lambda_csl * loss_csl + args.lambda_ssl * loss_ssl
                
                losses_ce.append(args.lambda_ce * loss_ce.numpy())
                losses_csl.append(args.lambda_csl * loss_csl.numpy())
                losses_ssl.append(args.lambda_ssl * loss_ssl.numpy())
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            if np.mod(i_iter+1, args.num_samples_per_image//args.batch_size_train) == 0:
                print('Epoch: {0:3d}, iter = {1:5d}, ce = {2:.4f}, csl = {3:.4f}, ssl = {4:.4f}'.format(epoch+1, i_iter+1, np.mean(np.array(losses_ce)), np.mean(np.array(losses_csl)), np.mean(np.array(losses_ssl))))
                losses_ce = []
                losses_csl = []
                losses_ssl = []
        
        # Test on epoch end
        current_best_mean_dice = test(model, img_path=args.test_img_path, label_path=args.test_label_path, subject_list=args.list_test_image_ids, patch_size=args.patch_size, batch_size=args.batch_size_test, numClasses=args.num_classes, epoch=epoch, current_best=current_best_mean_dice)
        
if __name__ == '__main__':

    Trainer()
