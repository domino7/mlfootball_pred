from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import numpy as np
import tensorflow as tf

import pandas as pd
import sklearn.model_selection as skl

#TODO: It's using bad files from parent directory, should use this directory folder
'''
Only 4 nparrays are needed but you don't get multhithreading and you can have heap overload when you load too much data

'''
# reset everything to rerun in jupyter
tf.reset_default_graph()

# config

batch_size = 10
learning_rate = 0.003
training_epochs = 120


ML_DATASET = "learning_vectors03.csv"
"""
X = pd.read_csv(ML_DATASET, usecols=['League_id', 'B365H' , 'B365D', 'B365A',
                                        'H_Speed', 'H_Pass' , 'H_Shoot' , 'H_Pressure',
                                        'H_chPass', 'H_chCross', 'H_dAggr', 'H_dWidth', 'A_Speed',
                                        'A_Pass','A_Shoot', 'A_Pressure', 'A_chPass', 'A_chCross',
                                        'A_dAggr', 'A_dWidth', 'H_age', 'A_age', 'H_TMV', 'A_TMV'])
"""
X = pd.read_csv(ML_DATASET, usecols=['H_age', 'A_age', 'H_TMV', 'A_TMV'])
IN_FEATURES = 4
logs_path = "results/compareSimple/004-transfermarkt"


Y = pd.read_csv(ML_DATASET, usecols=['Result'])



dataset = skl.train_test_split(X, Y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = [i.as_matrix() for i in dataset]

oneHot = []
for label in y_train:
    if label == 0:
        oneHot.append([1, 0, 0])
    if label == 1:
        oneHot.append([0, 1, 0])
    if label == 2:
        oneHot.append([0, 0, 1])
y_train = np.asarray(oneHot)

oneHot = []
for label in y_test:
    if label == 0:
        oneHot.append([1, 0, 0])
    if label == 1:
        oneHot.append([0, 1, 0])
    if label == 2:
        oneHot.append([0, 0, 1])
y_test = np.asarray(oneHot)


#Now Machine Learning staff:

# input images
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, IN_FEATURES], name="x-input")
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 3], name="y-input")

# model parameters will change during training so we use tf.Variable
with tf.name_scope("weights"):
    W = tf.Variable(tf.truncated_normal([IN_FEATURES, 3], stddev=0.1))

# bias
with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([3]))

# implement model
with tf.name_scope("softmax"):
    # y is our prediction
    y = tf.nn.softmax(tf.matmul(x, W) + b)

# specify cost function
with tf.name_scope('cross_entropy'):
    # this is our cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-16, 1.0)), reduction_indices=[1]))

# specify optimizer
with tf.name_scope('train'):
    # optimizer is an "operation" which we can execute in a session
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a summary for our cost and accuracy
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

# merge all summaries into a single "operation" which we can execute in a session
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # variables need to be initialized before we can use them
    sess.run(tf.global_variables_initializer())

    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # perform training cycles
    for epoch in range(training_epochs):

        # number of batches in one epoch
        batch_count = int(120/ batch_size) #120 - liczba danych uczacych

        for i in range(batch_count):
            #batch_x = X_train[i*batch_size:(i+1)*batch_size]
            #batch_y = y_train[i*batch_size:(i+1)*batch_size]
            batch_x = X_train
            batch_y = y_train
            # perform the operations we defined earlier on batch
            _, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y_: batch_y})

            # write log
            writer.add_summary(summary, epoch * batch_count + i)

        if epoch % 5 == 0:
            print("Epoch: ", epoch)
        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={x: X_test, y_: y_test}))

print("done")
