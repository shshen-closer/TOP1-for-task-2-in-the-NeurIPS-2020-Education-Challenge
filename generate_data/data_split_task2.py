import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from sklearn.model_selection import KFold,StratifiedKFold
#df_train = pd.read_csv('train_data/train_task_1_2.csv')


random.seed = 10

seq_len = 10


train_q_a = np.load("q_a_train.npy", allow_pickle=True)
#random.shuffle(train_q_a)
kfold = KFold(n_splits=10, shuffle=False)

count = 1
for (train_index, valid_index) in kfold.split(train_q_a):
    print(count)
    train_data = train_q_a[train_index]
    valid_data = train_q_a[valid_index]
    np.random.shuffle(train_data)
    
    train_q = []
    train_a = []
    train_av = []
    
    valid_q = []
    valid_a = []
    valid_av = []
    
    
    for tt in train_data:
        train_q.append(tt[0])
        train_a.append(tt[1])
        train_av.append(tt[2])
    np.save("train_q_"+ str(count) + ".npy",np.array(train_q))
    np.save("train_a_"+ str(count) + ".npy",np.array(train_a))
    np.save("train_av_"+ str(count) + ".npy",np.array(train_av))
    for tt in valid_data:
        valid_q.append(tt[0])
        valid_a.append(tt[1])
        valid_av.append(tt[2])
    np.save("valid_q_"+ str(count) + ".npy",np.array(valid_q))
    np.save("valid_a_"+ str(count) + ".npy",np.array(valid_a))
    np.save("valid_av"+ str(count) + ".npy",np.array(valid_av))
    
    count+=1


test_q_a = np.load("q_a_test.npy", allow_pickle=True)

test_all =  pd.read_csv('../data_preprocess/submission_task_1_2_new.csv')
test_all = np.array(test_all)

se_dict = {}

idx = 0
for line in test_all:
    se_dict[line[-3]] = idx
    idx+=1

test_q_a_new = []
for item in test_q_a:
    test_q_a_new.append([item[0],item[1],item[2], se_dict[item[3]]])

test_q_a_sorted = sorted(test_q_a_new, key = lambda x:x[-1])


test_q = []
test_a = []
test_av = []
for tt in test_q_a_sorted:
    test_q.append(tt[0])
    test_a.append(tt[1])
    test_av.append(tt[2])
    

np.save("test_q.npy",np.array(test_q))
np.save("test_a.npy",np.array(test_a))
np.save("test_av.npy",np.array(test_av))
