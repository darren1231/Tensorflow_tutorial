# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:37:40 2017

@author: darren
"""

import tensorflow as tf
import numpy as np
import os

store_network_path="temp/my_networks/"
if os.path.exists('temp'):
    pass
else:
    os.makedirs(store_network_path)
    
desired_input = np.random.rand(100)
desired_output = 3*desired_input+5

weights = tf.Variable(tf.random_uniform([1],-1,1))
biases = tf.Variable(tf.zeros([1]))

net_output = desired_input*weights + biases

loss = tf.reduce_mean(tf.square(desired_output-net_output))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# saving and loading networks
saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

checkpoint = tf.train.get_checkpoint_state(store_network_path)

if checkpoint and checkpoint.model_checkpoint_path:
    print ("checkpoint status:",checkpoint)
    print ("model checkpoint path:",checkpoint.model_checkpoint_path)
    
    saver.restore(sess,checkpoint.model_checkpoint_path)
    print ("Load old weight successful")
else:
    print ("Couldn't find old weights")


for i in range(200):
    sess.run(optimizer)
    if i%20==0:
        print(i,sess.run(weights),sess.run(biases),sess.run(loss))
        saver.save(sess, store_network_path, global_step = i)





