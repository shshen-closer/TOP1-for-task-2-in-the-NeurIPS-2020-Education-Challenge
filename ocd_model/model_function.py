# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:14:05 2019

@author:shshen
"""

import tensorflow as tf
import numpy as np





def cnn_block(x, filter1, filter2, kk, drop_rate, is_training):
    
    o1 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=is_training))

    res_w1 = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()([kk, filter1, filter2]),dtype=tf.float32)
    res_b1 = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()([filter2]),dtype=tf.float32)
    
    o2 = tf.nn.conv1d(o1, res_w1, 1, padding='SAME') + res_b1


    return o2

def Resnet(x, drop_rate, span, is_training):

    with tf.name_scope("block1"):
        for i in range(3):
            x = cnn_block(x,256,256, span, drop_rate, is_training)
    return x

def multi_span(x, drop_rate, is_training):
    x = tf.compat.v1.layers.dense(x, 256)
    filter_sizes = [2,3,4]
    all_output = []
    for filter_size in filter_sizes:
        with tf.name_scope("conv-filter{0}".format(filter_size)):
            output_i = Resnet(x, drop_rate, filter_size, is_training)
        all_output.append(tf.expand_dims(output_i, axis = -1))
    x = tf.concat(all_output, axis=-1)
    x = tf.reduce_mean(x, axis = -1)
        
    print(np.shape(x))
    return x
  