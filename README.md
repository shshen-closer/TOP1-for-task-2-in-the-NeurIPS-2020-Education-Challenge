# TOP1-for-task-2-in-the-NeurIPS-2020-Education-Challenge

This is the top1 solution in task 2 of the NeurIPS 2020 Education Challenge

requirements: 

Linux, 

Python 3.7, 

tensorflow 2.3.0, 

Scikit-learn 0.23.2


Firstly, download the data from the competiton website, then run following code in order:

1. data preprocess

question_meata_change.py: reindex the knowledge tags (knowledge concepts)

answer_meta_change.py: change time to the form of continuous integers timestamp

task_12_change.py: add two new columns, timestamp and quizid, to the traning data

2. generate data

task2_data_quiz.py: generate the data for traning and testing. The answered records are sorted by quizid and timestamp, sequence length is 10

data_split_task2.py: split 10-fold traing data

3. ocd_model

model training: python train.py kfold batch_size

model testing: python test.py kfold model_name

The model architecture is below:

![image](https://github.com/shshen-closer/TOP1-for-task-2-in-the-NeurIPS-2020-Education-Challenge/blob/main/OCD_framework.png)
