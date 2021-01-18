# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import logging
import codecs
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split



def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger

def read_data(seq_len):
    skill = {}
    with codecs.open('input_data_new.json', 'r', encoding = 'utf-8') as fi:
        for line in fi:
            temp = json.loads(line)
            skill[temp['testid']] =  temp['labels_index']
          
    q_a = []
    maxnum_skill = 0
    with codecs.open('learning_record_over50.json', 'r', encoding = 'utf-8') as fi:
        for line in fi:
            item = json.loads(line)
            for temp in item.values():
                ll = 0
                if ll + seq_len < len(temp):
                    a = np.zeros((seq_len,120))
                    b = np.zeros(seq_len)
                    for i in range(seq_len):
                        if temp[i+ll][0][1:-1] not in skill.keys():
                            ll += seq_len
                            break
                        if max(skill[temp[i+ll][0][1:-1]]) > maxnum_skill:
                            maxnum_skill = max(skill[temp[i+ll][0][1:-1]])
                        for s in skill[temp[i+ll][0][1:-1]]:
                            a[i, s] = 1
                            
                        b[i] = (int(float(temp[i+ll][1]) > 0.5))
                    ll += seq_len
                    if i == seq_len -1 :
                        q_a.append([a, b])
    
    random.shuffle(q_a)
    print(len(q_a))
    train, test  = train_test_split(q_a,test_size=0.05, random_state=0)
    train_q = []
    train_a = []
    test_q = []
    test_a = []
    for tt in train:
        train_q.append(tt[0])
        train_a.append(tt[1])
    for tt in test:
        test_q.append(tt[0])
        test_a.append(tt[1])
    return np.array(train_q), np.array(train_a), np.array(test_q), np.array(test_a), maxnum_skill + 1, seq_len

def read_textdata(seq_len):
    skill = {}
    with codecs.open('pre_feature', 'r', encoding = 'utf-8') as fi:
        for line in fi:
            temp = line.strip().split('\t')
            iid = temp[0]
            features = temp[1][1:-1].split(' ')
            features = [float(i) for i in features]
            skill[iid] =  features
          
    q_a = []
    with codecs.open('learning_record_over50.json', 'r', encoding = 'utf-8') as fi:
        for line in fi:
            item = json.loads(line)
            for temp in item.values():
                ll = 0
                if ll + seq_len < len(temp):
                    a = np.zeros((seq_len,100))
                    b = np.zeros(seq_len)
                    for i in range(seq_len):
                        if temp[i+ll][0][1:-1] not in skill.keys():
                            ll += seq_len
                            break
                        
                        a[i] = skill[temp[i+ll][0][1:-1]]
                            
                        b[i] = (int(float(temp[i+ll][1]) > 0.5))
                    ll += seq_len
                    if i == seq_len -1 :
                        q_a.append([a, b])
    
    random.shuffle(q_a)
    print(len(q_a))
    train, test  = train_test_split(q_a,test_size=0.05, random_state=0)
    train_q = []
    train_a = []
    test_q = []
    test_a = []
    for tt in train:
        train_q.append(tt[0])
        train_a.append(tt[1])
    for tt in test:
        test_q.append(tt[0])
        test_a.append(tt[1])
    return np.array(train_q), np.array(train_a), np.array(test_q), np.array(test_a), 100, seq_len

