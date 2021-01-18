# -*- coding:utf-8 -*-
__author__ = 'shshen'

import numpy as np
import tensorflow as tf

from model_function import *
#from densenet import *
#from resnet import *
class OCD(object):

    def __init__(self, batch_size, num_skills, hidden_size):
        
        self.batch_size = batch_size = batch_size
        self.hidden_size  = hidden_size
        self.num_skills =  num_skills


        
        self.input_id = tf.compat.v1.placeholder(tf.int32, [None, 9], name="input_id")
        self.x_answer = tf.compat.v1.placeholder(tf.float32, [None, 9,4], name="x_answer")
        self.valid_id = tf.compat.v1.placeholder(tf.int32, [None,], name="valid_id")
        self.target_correctness = tf.compat.v1.placeholder(tf.float32, [None,4], name="target_correctness")
        
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        
        self.global_step = tf.compat.v1.Variable(0, trainable=False, name="Global_Step")
        self.initializer=tf.compat.v1.keras.initializers.VarianceScaling()

        self.skill_w = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()([27613,self.num_skills]),dtype=tf.float32, trainable=True, name = 'skill_w')

        input_id = tf.nn.embedding_lookup(self.skill_w, self.input_id)
        skill = input_id
        
        valid_id = tf.nn.embedding_lookup(self.skill_w, self.valid_id)
        next_skill =valid_id


        x_answer = tf.tile(self.x_answer, [1,1,10])
        input_data = tf.concat([skill ,x_answer],axis = -1)
        input_data = tf.compat.v1.layers.dense(input_data, units = 256)
        #input_data = tf.nn.relu(input_data) 
        input_data = tf.nn.dropout(input_data, 0.5)

        outputs = multi_span(input_data, self.dropout_keep_prob, self.is_training)

        alpha = tf.matmul(skill,  tf.expand_dims(next_skill, axis = -1))
        alpha = tf.reshape(alpha, [-1, 9])
        alpha = tf.nn.softmax(alpha)
        outputs = tf.matmul(tf.transpose(outputs, [0,2,1]), tf.expand_dims(alpha, axis = -1))
        outputs = tf.reshape(outputs, [-1, self.hidden_size])
        
        outputs = tf.compat.v1.layers.dense(outputs, units = self.num_skills)
        outputs = tf.nn.relu(outputs)
        
        print(np.shape(next_skill))
        outputs = tf.concat([next_skill, outputs],axis = -1)
        
        outputs = tf.compat.v1.layers.dense(outputs, units = 128) 
        outputs = tf.nn.relu(outputs)   
       # outputs = tf.nn.dropout(outputs, self.dropout_keep_prob) 


        self.logits = tf.compat.v1.layers.dense(outputs, units = 4)
        print('aa')
        print(np.shape(self.logits))



        #make prediction
        self.pred = tf.sigmoid(self.logits, name="pred")

        # loss function
        #self.loss = tf.reduce_sum(tf.abs(self.logits - self.target_correctness))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.target_correctness), name="losses") 
        self.cost = self.loss
