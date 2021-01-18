import numpy as np
import pandas as pd
import time, datetime

#df_train = pd.read_csv('train_data/train_task_1_2.csv')


subject =  pd.read_csv('../public_data/metadata/subject_metadata.csv')  
subject = np.array(subject)
sss = 0
related_kill = {}


question_all =  pd.read_csv('../public_data/metadata/question_metadata_task_1_2.csv')


a = []
for x in question_all['SubjectId']:
    
    a.extend(eval(x))
a = set(a)
a = sorted(a)

subject_dict = {}
count = 0
for item in a:
    subject_dict[item] = count
    count += 1


question_all['knowledge_concept'] =  question_all['SubjectId'].apply(lambda x:[subject_dict[int(i)] for i in eval(x)])

question_new = question_all.drop(['SubjectId'], axis = 1)

question_new.to_csv('question_metadata_processed_1_2.csv',index=False)