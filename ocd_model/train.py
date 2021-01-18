# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import logging
import random
import pandas as pd
import tensorflow as tf
from datetime import datetime
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import metrics
from model_OCD import OCD

import checkmate as cm
import data_helpers as dh


# Parameters
# ==================================================

logger = dh.logger_fn("tflog", "logs/training_kfold_{0}_{1}_time_{2}.log".format(sys.argv[1], sys.argv[0], int(time.time())))

kfold= int(sys.argv[1])
batch_size = int(sys.argv[2])

tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.5, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 256, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", batch_size , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("epochs", 6, "Number of epochs to train for.")
tf.compat.v1.flags.DEFINE_integer("kfold", kfold, "Number of epochs to train for.")

tf.compat.v1.flags.DEFINE_integer("decay_steps",4, "how many steps before decay learning rate. (default: 500)")
tf.compat.v1.flags.DEFINE_float("decay_rate", 0.3, "Rate of decay for learning rate. (default: 0.95)")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 1000)")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100



def train():
    """Training model."""

    logger.info("Loading data...")

    logger.info("Training data processing...")
    max_num_skills, kfold = 388, FLAGS.kfold
    train_q = np.load("../generate_data/train_q_" + str(kfold) + ".npy", allow_pickle=True)
    train_a = np.load("../generate_data/train_av_" + str(kfold) + ".npy", allow_pickle=True)
    train_c = np.load("../generate_data/train_a_" + str(kfold) + ".npy", allow_pickle=True)
    valid_q = np.load("../generate_data/valid_q_" + str(kfold) + ".npy", allow_pickle=True)
    valid_a = np.load("../generate_data/valid_av" + str(kfold) + ".npy", allow_pickle=True)
    valid_c = np.load("../generate_data/valid_a_" + str(kfold) + ".npy", allow_pickle=True)


    ques_kc = pd.read_csv('../data_preprocess/question_metadata_processed_1_2.csv')
    ques_kc = np.array(ques_kc)
    ques_dict = {}
    for items in ques_kc:
        ques_dict[int(items[0])] = eval(items[1])

    print(len(train_q))

    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            ocd = OCD(
                batch_size = FLAGS.batch_size,
                num_skills = max_num_skills,
                hidden_size = FLAGS.hidden_size,
                )

            # Define training procedure
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=ocd.global_step, decay_steps=(len(train_q)//FLAGS.batch_size +1) * FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
               # learning_rate = tf.train.piecewise_constant(FLAGS.epochs, boundaries=[7,10], values=[0.005, 0.0005, 0.0001])
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(ocd.loss, global_step=ocd.global_step, name="train_op")

            # Output directory for models and summaries

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            logger.info("Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.compat.v1.summary.scalar("loss", ocd.loss)

            # Train summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.compat.v1.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.compat.v1.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())



            current_step = sess.run(ocd.global_step)

            def train_step(input_id, x_answer,valid_id, target_correctness):
                """A single training step"""

                feed_dict = {
                    ocd.input_id:input_id,
                    ocd.x_answer: x_answer,
                    ocd.valid_id: valid_id,
                    ocd.target_correctness: target_correctness,
                    ocd.dropout_keep_prob: FLAGS.keep_prob,
                    ocd.is_training: True
                }
                _, step, summaries, pred, loss = sess.run(
                    [train_op, ocd.global_step, train_summary_op, ocd.pred, ocd.loss], feed_dict)

                acc = np.mean(np.equal(np.argmax(target_correctness, -1), np.argmax(pred, -1)).astype(int))
                print("step {0}: loss {1:g} acc:{2:g} ".format(step,loss, acc))
                logger.info("step {0}: loss {1:g} acc:{2:g} ".format(step,loss, acc))
                train_summary_writer.add_summary(summaries, step)
                return pred

            def validation_step( input_id, x_answer,valid_id, target_correctness):
                """Evaluates model on a validation set"""

                feed_dict = {
                    ocd.input_id:input_id,
                    ocd.x_answer: x_answer,
                    ocd.valid_id: valid_id,
                    ocd.target_correctness: target_correctness,
                    ocd.dropout_keep_prob: 0.0,
                    ocd.is_training: False
                }
                step, summaries, pred, loss = sess.run(
                    [ocd.global_step, validation_summary_op, ocd.pred, ocd.loss], feed_dict)
                validation_summary_writer.add_summary(summaries, step)
                return pred
            # Training loop. For each batch...

            def one_hot_answer(input_data,length):
                output = np.zeros((length,10,4),  dtype=float)
                for item in range(len(input_data)):
                    for iii in range(len(input_data[item])):
                        if input_data[item][iii] < 4:
                            output[item,iii,input_data[item][iii]] = 1
                return output

            run_time = []
            m_acc = 0
            for iii in range(FLAGS.epochs):

                a=datetime.now()
                data_size = len(train_q)
                index = FLAGS.batch_size
                batch = 0
                actual_labels = []
                pred_labels = []
                while(index*batch+FLAGS.batch_size < data_size):
                    question_id = train_q[index*batch : index*(batch+1)]
                    np.random.seed(iii*100)
                    np.random.shuffle(question_id)


                    answer = train_a[index*batch : index*(batch+1)]
                    np.random.seed(iii*100)
                    np.random.shuffle(answer)
                    answer = one_hot_answer(answer, FLAGS.batch_size)
                                        
                    input_id = question_id[:,:9]
                    valid_id = question_id[:, -1]

                    x_answer = answer[:,:9, :]

                    target_correctness = answer[:,-1,:]
                    
                    actual_labels.extend(np.argmax(target_correctness, -1))
                   # print(np.reshape(target_correctness, [-1]))
                    pred = train_step(input_id, x_answer, valid_id, target_correctness)
                    pred_labels.extend(np.argmax(pred, -1))
                  #  print(np.argmax(pred, -1))
                    
                    
                    current_step = tf.compat.v1.train.global_step(sess, ocd.global_step)
                    batch += 1

                b=datetime.now()
                e_time = (b-a).total_seconds()
                run_time.append(e_time)
               # rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
               # fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
                #auc = metrics.auc(fpr, tpr)
                #calculate r^2
                #r2 = r2_score(actual_labels, pred_labels)
               # pred_score = np.greater(pred_labels,0.5)
                pred_score = np.array(pred_labels)
                pred_score = np.equal(np.array(actual_labels), pred_score)
                acc = np.mean(pred_score.astype(int))
                logger.info("epochs {0}: acc {1:g}  ".format((iii +1), acc))

                if((iii+1) % FLAGS.evaluation_interval == 0):
                    logger.info("\nEvaluation:")

                    index = FLAGS.batch_size
                    data_size = len(valid_q)
                    batch = 0
                    actual_labels = []
                    pred_labels = []
                    while(index*batch+FLAGS.batch_size < data_size):
                        question_id = valid_q[index*batch : index*(batch+1)]
                        answer = valid_a[index*batch : index*(batch+1)]
                        answer = one_hot_answer(answer, FLAGS.batch_size)
                                                
                        input_id = question_id[:,:9]
                        valid_id = question_id[:, -1]

                        x_answer = answer[:,:9,:]

                        target_correctness = answer[:,-1,:]
                        actual_labels.extend(np.argmax(target_correctness, -1))

                        batch += 1
                        #print(ability)
                        pred = validation_step( input_id, x_answer,valid_id, target_correctness)
                        pred_labels.extend(np.argmax(pred, -1))

                        current_step = tf.compat.v1.train.global_step(sess, ocd.global_step)

                 #   rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                   # fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
                   # auc = metrics.auc(fpr, tpr)
                    #calculate r^2
                  #  r2 = r2_score(actual_labels, pred_labels)
                   # pred_score = np.greater(pred_labels,0.5)
                    pred_score = np.array(pred_labels)
                    pred_score = np.equal(np.array(actual_labels), pred_score)
                    acc = np.mean(pred_score.astype(int))

                    logger.info("VALIDATION {0}: acc {1:g} ".format((iii +1)/FLAGS.evaluation_interval,acc))
                    print("VALIDATION {0}: acc {1:g} ".format((iii +1)/FLAGS.evaluation_interval,acc))


                    if acc > m_acc:
                        m_acc = acc

                    best_saver.handle(acc, sess, current_step)
                if ((iii+1) % FLAGS.checkpoint_every == 0):
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))

                logger.info("Epoch {0} has finished!".format(iii + 1))

            logger.info("running time analysis: epoch{0}, avg_time{1}".format(len(run_time), np.mean(run_time)))
            logger.info("max: acc{0:g} ".format(m_acc))
            with open('results.txt', 'a') as fi:
                fi.write( ':\n')
                fi.write("max: acc {0:g}  ".format(m_acc))
                fi.write('\n')
    logger.info("Done.")


if __name__ == '__main__':
    train()
