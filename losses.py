import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

def CSL(gt, semantic, alpha=0.25, gamma=2.0):
    
    epsilon = K.epsilon()
    semantic = tf.clip_by_value(semantic, epsilon, 1. - epsilon)
    
    ce = - ( gt * tf.math.log(semantic) + (1. - gt) * tf.math.log(1. - semantic) )
    
    p_t = (gt * semantic) + ((1 - gt) * (1 - semantic))
    alpha_factor = 1.0
    modulating_factor = 1.0
    
    alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
    alpha_factor = gt * alpha + (1 - gt) * (1 - alpha)
    
    gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
    modulating_factor = tf.math.pow((1.0 - p_t), gamma)
    
    # compute the final loss and return
    return tf.reduce_mean(tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=[-1, -2]))

def SSL(y_true, y_pred, num_classes=2, dilations=[1, 2, 3], kernelSize=3, margin=2):
    
    epsilon = K.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # patches_true #
    patches_true_list = []
    for class_id in range(num_classes):
        patches_true_class_id_list = []
        y_true_id = y_true[:, class_id:class_id+1, :, :, :]
        for dilation_id in range(len(dilations)):
            dr = [1, 1] + [dilations[dilation_id],]*3
            
            w = np.eye(27).reshape((3, 3, 3, 1, 27))
            w = w[:, :, :, :, [4, 10, 12, 14, 16, 22]]
            weights=tf.stop_gradient(tf.convert_to_tensor(w, tf.float32))
            
            padSize = kernelSize // 2 * dilations[dilation_id]
            padded_true_id = tf.pad(y_true_id, tf.constant([[0, 0], [0, 0], [padSize, padSize,], [padSize, padSize], [padSize, padSize]]), mode='SYMMETRIC')
            patches_true_id= tf.nn.conv3d(padded_true_id, weights, strides=[1, 1, 1, 1, 1], data_format='NCDHW', padding='VALID', dilations=dr)
            patches_true_class_id_list.append(tf.expand_dims(patches_true_id, axis=1))
        
        patches_true_list.append(tf.concat(patches_true_class_id_list, axis=2))
    
    patches_true = tf.concat(patches_true_list, axis=1)
    
    # patches_pred #
    patches_pred_list = []
    for class_id in range(num_classes):
        patches_pred_class_id_list = []
        y_pred_id = y_pred[:, class_id:class_id+1, :, :, :]
        for dilation_id in range(len(dilations)):
            dr = [1, 1] + [dilations[dilation_id],]*3
            
            w = np.eye(27).reshape((3, 3, 3, 1, 27))
            w = w[:, :, :, :, [4, 10, 12, 14, 16, 22]]
            weights=tf.stop_gradient(tf.convert_to_tensor(w, tf.float32))
            
            padSize = kernelSize // 2 * dilations[dilation_id]
            padded_pred_id = tf.pad(y_pred_id, tf.constant([[0, 0], [0, 0], [padSize, padSize,], [padSize, padSize], [padSize, padSize]]), mode='SYMMETRIC')
            patches_pred_id= tf.nn.conv3d(padded_pred_id, weights, strides=[1, 1, 1, 1, 1], data_format='NCDHW', padding='VALID', dilations=dr)
            patches_pred_class_id_list.append(tf.expand_dims(patches_pred_id, axis=1))
        
        patches_pred_list.append(tf.concat(patches_pred_class_id_list, axis=2))
    
    patches_pred = tf.stop_gradient(tf.concat(patches_pred_list, axis=1))
    #patches_pred = tf.concat(patches_pred_list, axis=1)
    
    same_or_not = tf.reduce_sum( patches_true * tf.expand_dims(y_true, axis=2), axis=1 )
    
    y_pred = tf.expand_dims(y_pred, axis=2)
    kl_distance = tf.reduce_sum( y_pred * tf.math.log(y_pred / patches_pred), axis=1)
    kl_distance = tf.clip_by_value(kl_distance, 0.0, 1e2)
    
    loss = tf.where(tf.greater(same_or_not, 0.5), kl_distance, tf.maximum(margin-kl_distance, 0))
    
    return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(loss, axis=1), axis=[-1, -2, -3]))

def custom_categorical_crossentropy(gt, pred):
    gt = tf.cast(gt, tf.float32)
    # manual computation of crossentropy
    epsilon = K.epsilon()
    pred = tf.clip_by_value(pred, epsilon, 1. - epsilon)
    return - tf.reduce_mean(tf.reduce_sum(gt * tf.math.log(pred), axis=1))
