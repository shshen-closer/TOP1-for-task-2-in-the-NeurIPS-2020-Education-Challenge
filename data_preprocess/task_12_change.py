import numpy as np
import pandas as pd
import time, datetime

#df_train = pd.read_csv('train_data/train_task_1_2.csv')

answer_sorted = pd.read_csv('answer_metadata_sorted_1_2.csv')
answer_sorted = np.array(answer_sorted)
answer_dict = {}
count = 0
quiz_dict = {}
for item in answer_sorted:
    count+=1
    if str(item[2]) == 'nan':
        print('nan')
        continue
    if str(item[0]) == 'nan':
        print(count)
        continue
    answer_dict[int(item[0])] = float(item[4])
    quiz_dict[int(item[0])] = int(item[2])

    
train_all =  pd.read_csv('../public_data/train_data/submission_task_1_2.csv')

train_all['timestamp'] =  train_all['AnswerId'].apply(lambda x:answer_dict[x])
train_all['quizid'] =  train_all['AnswerId'].apply(lambda x:quiz_dict[x])

#train_new = train_all.drop(['AnswerId'], axis = 1)

train_all.to_csv('submission_task_1_2_new.csv',index=False)

train_all =  pd.read_csv('../public_data/train_data/train_task_1_2.csv')

train_all['timestamp'] =  train_all['AnswerId'].apply(lambda x:answer_dict[x])
train_all['quizid'] =  train_all['AnswerId'].apply(lambda x:quiz_dict[x])

#train_new = train_all.drop(['AnswerId'], axis = 1)

train_all.to_csv('train_task_1_2_new.csv',index=False)