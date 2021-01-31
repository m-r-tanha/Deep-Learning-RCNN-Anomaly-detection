# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:54:52 2021

@author: Mohammad.Tanhatalab
"""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, normalize, minmax_scale
#from tensorflow.contrib import rnn
#from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from pandas import ExcelWriter
import xlsxwriter
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.reset_default_graph()


def RCNN (data):
    import tensorflow.compat.v1 as tf
    import keras
    tf.disable_v2_behavior() 
    tf.reset_default_graph()
    Normalized_X=scale(data, axis=1)
    Mi=Normalized_X.min()   
    Normalized_X=(-1*Mi)+Normalized_X
    
    
    # Y_train = np_utils.to_categorical(Y_train)
    # Y_test = np_utils.to_categorical(Y_test)
    
    # Y_train = np.delete(Y_train , 0, 1)
    # Y_test = np.delete(Y_test , 0, 1)
    # Training Parametersznum_steps = 30
    #num_steps=500
    batch_size = 120
    #display_step = 1
    strides = 1
    k = 1
    
    # Network Parameters
    num_input = 31  #  data input (img shape: 28*28)
    num_hidden = 100
    num_classes = 6  #  total classes (0-9 digits)
    dropout = 0.7  # Dropout, probability to keep units
    
    # tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    
    is_training = tf.placeholder(tf.bool, name='MODE')
    
    
    # Store layers weight & bias
    # The first three convolutional layer
    w_c_1 = tf.Variable(tf.random_normal([1, 3, 1, 28]))
    w_c_2 = tf.Variable(tf.random_normal([1, 3, 28, 56]))
    w_c_3 = tf.Variable(tf.random_normal([1, 3, 56, 112]))
    b_c_1 = tf.Variable(tf.zeros([28]))
    b_c_2 = tf.Variable(tf.zeros([56]))
    b_c_3 = tf.Variable(tf.zeros([112]))
    
    # The second three convolutional layer weights
    w_c_4 = tf.Variable(tf.random_normal([1, 3, 112, 224]))
    w_c_5 = tf.Variable(tf.random_normal([1, 3, 224, 448]))
    w_c_6 = tf.Variable(tf.random_normal([1, 3, 448, 896]))
    b_c_4 = tf.Variable(tf.zeros([224]))
    b_c_5 = tf.Variable(tf.zeros([448]))
    b_c_6 = tf.Variable(tf.zeros([896]))
    
    # Fully connected weight
    w_f_1 = tf.Variable(tf.random_normal([1 * 31 * 896, 1792])) # fully connected, 1*3*896 inputs, 2048 outputs
    w_f_2 = tf.Variable(tf.random_normal([1792, 896]))
    w_f_3 = tf.Variable(tf.random_normal([896, 448]))
    b_f_1 = tf.Variable(tf.zeros([1792]))
    b_f_2 = tf.Variable(tf.zeros([896]))
    b_f_3 = tf.Variable(tf.zeros([448]))
    
    # output layer weight
    w_out = tf.Variable(tf.random_normal([448, num_classes]))
    b_out = tf.Variable(tf.zeros([num_classes]))
    
    #
    # Define model
    x = tf.reshape(X, shape=[-1, 1, 31, 1])
    
    
    # first layer convolution
    conv1 = tf.nn.conv2d(x, w_c_1, strides=[1, 1, 1, 1], padding='SAME') + b_c_1
    conv1 = tf.compat.v1.layers.batch_normalization(conv1, training=True, center=True, scale=False)
    conv1 = tf.nn.tanh(conv1)
    
    
    # second layer convolution
    conv2 = tf.nn.conv2d(conv1, w_c_2, strides=[1, strides, strides, 1], padding='SAME') + b_c_2
    conv2 = tf.compat.v1.layers.batch_normalization(conv2, training=True, center=True, scale=False)
    conv2 = tf.nn.tanh(conv2)
    conv2 = tf.nn.dropout(conv2, dropout)
    
    # third layer convolution
    conv3 = tf.nn.conv2d(conv2, w_c_3, strides=[1, strides, strides, 1], padding='SAME') + b_c_3
    conv3 = tf.compat.v1.layers.batch_normalization(conv3, training=True, center=True, scale=False)
    conv3 = tf.nn.tanh(conv3)
    
    # first Max Pooling (down-sampling)
    pool_1 = tf.nn.max_pool(conv3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    # fourth layer convolution
    conv4 = tf.nn.conv2d(pool_1, w_c_4, strides=[1, strides, strides, 1], padding='SAME') + b_c_4
    conv4 = tf.compat.v1.layers.batch_normalization(conv4, training=True, center=True, scale=False)
    conv4 = tf.nn.tanh(conv4)
    conv4 = tf.nn.dropout(conv4, dropout)
    
    # fifth layer convolution
    conv5 = tf.nn.conv2d(conv4, w_c_5, strides=[1, strides, strides, 1], padding='SAME') + b_c_5
    conv5 = tf.compat.v1.layers.batch_normalization(conv5, training=True, center=True, scale=False)
    conv5 = tf.nn.tanh(conv5)
    
    # sixth layer convolution
    conv6 = tf.nn.conv2d(conv5, w_c_6, strides=[1, strides, strides, 1], padding='SAME') + b_c_6
    conv6 = tf.compat.v1.layers.batch_normalization(conv6, training=True, center=True, scale=False)
    conv6 = tf.nn.tanh(conv6)
    conv6 = tf.nn.dropout(conv6, dropout)
    
    # second Max Pooling (down-sampling)
    pool_2 = tf.nn.max_pool(conv6, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    pool_2 = tf.reshape(pool_2, [-1,31,896])
    
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden,num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    def RNN(x, weights, biases):
      
        lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_hidden,reuse=tf.AUTO_REUSE)
        outputs, states = tf.compat.v1.nn.dynamic_rnn(lstm_cell, x , dtype=tf.float32)
        return tf.matmul(outputs[:,-1], weights['out']) + biases['out']
        
        logits = RNN(pool_2, weights, biases)
        prediction = tf.nn.softmax(logits)
    
    logits = RNN(pool_2, weights, biases)
    prediction = tf.nn.softmax(logits)
    
    
    saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    
    with tf.Session(config=config) as sess:
        
        saver.restore(sess,"./TF_ModelwBN_for_Paper4.ckpt")
        out=sess.run(prediction, feed_dict={X: Normalized_X})
    return out   


data_path=r"G:\Data Science Report\Raw Data\Main KPI for 33 days 2021.01.20.xlsx"
data_Thr = pd.read_excel(data_path,sheet_name="Thr")
data_Thr = data_Thr.fillna(0.0)
data_Thr.rename(index=data_Thr.SITE,inplace=True)
data_Thr.drop('SITE',axis=1,inplace=True)

data_Pay = pd.read_excel(data_path,sheet_name="Payload")
data_Pay = data_Pay.fillna(0.0)
data_Pay.rename(index=data_Pay.SITE,inplace=True)
data_Pay.drop('SITE',axis=1,inplace=True)

out_Paylaod = RCNN(data_Pay)
out_Paylaod=pd.DataFrame(out_Paylaod)
out_Paylaod.columns=['N','DS','SI','GD','SD','GI']

out_Thr = RCNN(data_Thr)
out_Thr =pd.DataFrame(out_Thr)
out_Thr.columns=['N','DS','SI','GD','SD','GI']

writer = pd.ExcelWriter('Site_Thr_Degr.xlsx')
out_Thr.to_excel(writer,'Thr_result')
out_Paylaod.to_excel(writer,'Payload_result')

data_Thr.to_excel(writer,'Thr_data')
data_Pay.to_excel(writer,'Payload_data')
writer.save()

