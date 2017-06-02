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



ML_DATASET = "learning_vectors03.csv"

X = pd.read_csv(ML_DATASET, usecols=['League_id', 'B365H' , 'B365D', 'B365A',
                                        'H_Speed', 'H_Pass' , 'H_Shoot' , 'H_Pressure',
                                        'H_chPass', 'H_chCross', 'H_dAggr', 'H_dWidth', 'A_Speed',
                                        'A_Pass','A_Shoot', 'A_Pressure', 'A_chPass', 'A_chCross',
                                        'A_dAggr', 'A_dWidth', 'H_age', 'A_age', 'H_TMV', 'A_TMV'])

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

print(X_train, X_test, y_train, y_test)

#Now Machine Learning staff:

logs_path = "/home/domgor/PycharmProjects/MLFootba/results/011-deep5r0.0005b15e35-wideLayers"
# config
K = 200
L = 100
M = 60
N = 30
pkeep = 0.75

batch_size =15
learning_rate = 0.0005
training_epochs = 35


# input images
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, 24], name="x-input")
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 3], name="y-input")

# model parameters will change during training so we use tf.Variable





#WARSTWA 1
with tf.name_scope("weights1"):
    W1 = tf.Variable(tf.truncated_normal([24, K], stddev=0.1))
# bias
with tf.name_scope("biases1"):
    b1 = tf.Variable(tf.zeros([K]))

#WARSTWA 2
with tf.name_scope("weights2"):
    W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
# bias
with tf.name_scope("biases2"):
    b2 = tf.Variable(tf.zeros([L]))

#WARSTWA 3
with tf.name_scope("weights3"):
    W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
# bias
with tf.name_scope("biases3"):
    b3 = tf.Variable(tf.zeros([M]))

#WARSTWA 4
with tf.name_scope("weights4"):
    W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
# bias
with tf.name_scope("biases4"):
    b4 = tf.Variable(tf.zeros([N]))

#WARSTWA 5
with tf.name_scope("weights5"):
    W5 = tf.Variable(tf.truncated_normal([N, 3], stddev=0.1))
# bias
with tf.name_scope("biases5"):
    b5 = tf.Variable(tf.zeros([3]))





# implement model

with tf.name_scope("relu1"):
    # yn is our prediction
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    #y1 = tf.nn.dropout(yf1, pkeep)

with tf.name_scope("relu2"):
    # yn is our prediction
    y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
    #y2 = tf.nn.dropout(yf2, pkeep)

with tf.name_scope("relu3"):
    # yn is our prediction
    y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
    #y3 = tf.nn.dropout(yf3, pkeep)

with tf.name_scope("relu4"):
    # yn is our prediction
    y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)
    #y4 = tf.nn.dropout(yf4, pkeep)


with tf.name_scope("softmax"):
    # y is our prediction
    y = tf.nn.softmax(tf.matmul(y4, W5) + b5)

# specify cost function
with tf.name_scope('cross_entropy'):
    # this is our cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))

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
        batch_count = int(4256/ batch_size) #120 - liczba danych uczacych

        for i in range(batch_count):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = y_train[i*batch_size:(i+1)*batch_size]

            # perform the operations we defined earlier on batch
            _, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y_: batch_y})

            # write log
            writer.add_summary(summary, epoch * batch_count + i)

        if epoch % 5 == 0:
            print("Epoch: ", epoch)
        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={x: X_test, y_: y_test}))

print("done")
