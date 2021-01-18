import numpy as np
import pandas as pd
import time, datetime

#df_train = pd.read_csv('train_data/train_task_1_2.csv')


answer_all =  pd.read_csv('../public_data/metadata/answer_metadata_task_1_2.csv')

answer_all['timestamp'] =  answer_all['DateAnswered'].apply(lambda x:time.mktime(time.strptime(x[:-4],'%Y-%m-%d %H:%M:%S')))

#answer_all.drop(['Confidence', 'GroupId', 'QuizId', 'SchemeOfWorkId', 'DateAnswered'], axis = 1, inplace=True)

answer_all.drop(['Confidence', 'DateAnswered'], axis = 1, inplace=True)

answer_sorted = answer_all.sort_values('timestamp')

answer_sorted = answer_sorted.reset_index(drop = True)

answer_sorted.to_csv('answer_metadata_sorted_1_2.csv',index=False)