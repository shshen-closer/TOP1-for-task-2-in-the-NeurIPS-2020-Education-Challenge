import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
#df_train = pd.read_csv('train_data/train_task_1_2.csv')



pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)

seq_len = 10

    
train_all =  pd.read_csv('../data_preprocess/train_task_1_2_new.csv')
test_all =  pd.read_csv('../data_preprocess/submission_task_1_2_new.csv')
train_all = train_all.drop(['CorrectAnswer'], axis = 1)
test_all = test_all.drop(['Unnamed: 0'], axis = 1)

train_id = set(np.array(train_all['UserId']))
test_id = set(np.array(test_all['UserId']))

all_data1 = pd.merge(train_all,test_all,how="outer")
all_data1['isc'] = all_data1['IsCorrect']
all_data1['isc'].fillna(2,inplace=True)
all_data = all_data1

order = ['AnswerId','UserId','QuestionId','IsCorrect','AnswerValue','timestamp','quizid','isc']
all_data = all_data[order]

ques_kc = pd.read_csv('../data_preprocess/question_metadata_processed_1_2.csv')
ques_kc = np.array(ques_kc)
ques_dict = {}
for item in ques_kc:
    ques_dict[float(item[0])] = eval(item[1])
    

length_train = []
q_a_train = []
q_a_test = []
for item in tqdm(test_id):

    idx = all_data[(all_data.UserId==item)].index.tolist()

    temp1 = all_data.iloc[idx]
    
    temp1 = temp1.sort_values(by=['quizid', 'timestamp'])
    
    temp1['IsCorrect'].fillna(2,inplace=True)
    temp1['AnswerValue'].fillna(5,inplace=True)
    temp_ori = np.array(temp1)

    temp_1=[]
    temp_2 = []
    temp_random = []
    reidx = 0

    # sorted by quizid and timestamp
    for i in range(len(temp_ori)-1):
        if temp_ori[i+1][-2] !=  temp_ori[i][-2]:
            temp_random.append(temp_ori[i])
            temp_quiz = []
            if len(temp_random) >=seq_len:
                for tt in range(len(temp_random) - 1):
                    if temp_random[tt+1][-3] - temp_random[tt][-3] > 3600*24*7:    # 7 days
                        temp_quiz.append(temp_random[tt])
                        if len(temp_quiz) >= seq_len:
                            temp_quiz_2 = [np.append(x, reidx) for x in temp_quiz]
                            reidx += 1
                            temp_1.append(temp_quiz_2)
                            temp_quiz = []
                        else:
                            temp_quiz_2 = [np.append(x, reidx) for x in temp_quiz]
                            reidx += 1
                            temp_2.append(temp_quiz_2)
                            temp_quiz = []
                    else:
                        temp_quiz.append(temp_random[tt])
                temp_quiz.append(temp_random[-1])
                if len(temp_quiz) >= seq_len:
                    temp_quiz_2 = [np.append(x, reidx) for x in temp_quiz]
                    reidx += 1
                    temp_1.append(temp_quiz_2)
                else:
                    temp_quiz_2 = [np.append(x, reidx) for x in temp_quiz]
                    reidx += 1
                    temp_2.append(temp_quiz_2)
            else:
                temp_quiz_2 = [np.append(x, reidx) for x in temp_random]
                reidx += 1
                temp_2.append(temp_quiz_2)
            temp_random = []
        else:
            temp_random.append(temp_ori[i])
    
    temp_random.append(temp_ori[-1])
    temp_quiz = []
    if len(temp_random) >=seq_len:
        for tt in range(len(temp_random) - 1):
            if temp_random[tt+1][-3] - temp_random[tt][-3] > 3600*24*7:
                temp_quiz.append(temp_random[tt])
                if len(temp_quiz) >= seq_len:
                    temp_quiz_2 = [np.append(x, reidx) for x in temp_quiz]
                    reidx += 1
                    temp_1.append(temp_quiz_2)
                    temp_quiz = []
                else:
                    temp_quiz_2 = [np.append(x, reidx) for x in temp_quiz]
                    reidx += 1
                    temp_2.append(temp_quiz_2)
                    temp_quiz = []
            else:
                temp_quiz.append(temp_random[tt])
        temp_quiz.append(temp_random[-1])
        if len(temp_quiz) >= seq_len:
            temp_quiz_2 = [np.append(x, reidx) for x in temp_quiz]
            reidx += 1
            temp_1.append(temp_quiz_2)
        else:
            temp_quiz_2 = [np.append(x, reidx) for x in temp_quiz]
            reidx += 1
            temp_2.append(temp_quiz_2)

    else:
        temp_quiz_2 = [np.append(x, reidx) for x in temp_random]
        reidx += 1
        temp_2.append(temp_quiz_2)

    temp = []
    for t1 in temp_2:
        temp.extend(t1)
    for t2 in temp_1:
        temp.extend(t2)
    temp = np.array(temp)
    temp_front = []

   # data process
    for i in range(len(temp)-1):
        if temp[i+1][-1] !=  temp[i][-1]:
            temp_front.append(temp[i])
            if len(temp_front) >= seq_len:
                for tt in range(len(temp_front)):
                    a = []              # questions
                    b = [0]*seq_len    #answer correctly or not
                    c = [0]*seq_len    #answer choice
                    
                    if tt < seq_len:
                        for ff in range(seq_len):
                            if ff<tt:
                                a.append(int(temp_front[ff][2]))
                                b[ff] = int(temp_front[ff][3])
                                c[ff] = int(temp_front[ff][4]) - 1
                            if ff>tt:
                                a.append(int(temp_front[ff][2]))
                                b[ff - 1] = int(temp_front[ff][3])
                                c[ff - 1] = int(temp_front[ff][4]) - 1
                        
                        a.append(int(temp_front[tt][2]))
                        b[seq_len-1] = int(temp_front[tt][3])
                        c[seq_len-1] = int(temp_front[tt][4]) - 1
            
                    if tt >= seq_len:
                        for ff in range(tt - seq_len+1, tt+1):
                            a.append(int(temp_front[ff][2]))
                            b[ff - tt + seq_len -1] = int(temp_front[ff][3])
                            c[ff - tt + seq_len -1] = int(temp_front[ff][4]) - 1
                    
                    if len(a)>seq_len:
                        print('iii')
                    if int(temp_front[tt][-2]) == 2:
                        q_a_test.append([a,b,c, temp_front[tt][0]])
                    else:
                        q_a_train.append([a,b, c,int(temp_front[tt][3]), int(temp_front[tt][4]) - 1])


                temp_front = []

        else:
            temp_front.append(temp[i])
    temp_front.append(temp[-1])
    if len(temp_front) >= seq_len:
        for tt in range(len(temp_front)):
            a = []
            b = [0]*seq_len
            c = [0]*seq_len
            
            if tt < seq_len:
                for ff in range(seq_len):
                    if ff<tt:
                        a.append(int(temp_front[ff][2]))
                        b[ff] = int(temp_front[ff][3])
                        c[ff] = int(temp_front[ff][4]) - 1
                    if ff>tt:
                        a.append(int(temp_front[ff][2]))
                        b[ff - 1] = int(temp_front[ff][3])
                        c[ff - 1] = int(temp_front[ff][4]) - 1
                
                a.append(int(temp_front[tt][2])) 
                b[seq_len-1] = int(temp_front[tt][3])
                c[seq_len-1] = int(temp_front[tt][4]) - 1
    
            if tt >= seq_len:
                for ff in range(tt - seq_len+1, tt+1):
                    a.append(int(temp_front[ff][2]))
                    b[ff - tt + seq_len -1] = int(temp_front[ff][3])
                    c[ff - tt + seq_len -1] = int(temp_front[ff][4]) - 1
            
            if len(a)>seq_len:
                print('iii')
            if int(temp_front[tt][-2]) == 2:
                q_a_test.append([a,b,c, temp_front[tt][0]])
            else:
                q_a_train.append([a,b, c,int(temp_front[tt][3]), int(temp_front[tt][4]) - 1])
        
                
    elif len(temp_front) > 0:
        for tt in range(len(temp) - len(temp_front), len(temp)):
            a = []
            b = [0]*seq_len
            c = [0]*seq_len
            for ff in range(len(temp) - seq_len,len(temp)):
                if ff<tt:
                    a.append(int(temp[ff][2]))
                    b[ff - len(temp) + seq_len] = int(temp[ff][3])
                    c[ff - len(temp) + seq_len] = int(temp[ff][4]) - 1
                if ff>tt:
                    a.append(int(temp[ff][2]))
                    b[ff - 1 - len(temp) + seq_len] = int(temp[ff][3])
                    c[ff - 1 - len(temp) + seq_len] = int(temp[ff][4]) - 1
                    
                    
            a.append(int(temp[tt][2]))
            b[seq_len-1] = int(temp[tt][3])
            c[seq_len-1] = int(temp[tt][4]) - 1
            
            if int(temp[tt][-2]) == 2:
                q_a_test.append([a,b,c, temp[tt][0]])
            else:
                q_a_train.append([a,b, c, int(temp[tt][3]), int(temp[tt][4]) - 1])

                   

print(len(q_a_test))
print(len(q_a_train))

np.save("q_a_train.npy",np.array(q_a_train))
np.save("q_a_test.npy",np.array(q_a_test))
print('complete')
