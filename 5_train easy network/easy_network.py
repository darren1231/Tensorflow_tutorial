# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:05:36 2017

@author: darren
"""

import tensorflow as tf
import numpy as np

desired_input = np.random.rand(100)
desired_output = 3*desired_input+5

weights = tf.Variable(tf.random_uniform([1],-1,1))
biases = tf.Variable(tf.zeros([1]))

net_output = desired_input*weights + biases

loss = tf.reduce_mean(tf.square(desired_output-net_output))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(200):
    sess.run(optimizer)
    if i%20==0:
        print(i,sess.run(weights),sess.run(biases),sess.run(loss))





