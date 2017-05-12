# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:27:06 2017

@author: darren
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


desired_input = tf.placeholder(tf.float32,[None,784])
desired_output = tf.placeholder(tf.float32,[None,10])

weights = tf.Variable(tf.zeros([784,10]))
biases = tf.Variable(tf.zeros([10]))
net_output = tf.nn.softmax(tf.matmul(desired_input,weights)+biases)

#loss_cross_entrop = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits \
#(labels=desired_output, logits=net_output))
loss_cross_entrop =tf.reduce_mean(-tf.reduce_sum(desired_output* \
tf.log(net_output),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_cross_entrop)

correct_prediction = tf.equal(tf.arg_max(desired_output,1),tf.arg_max(net_output,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in range(100):
    
    batch_desired_input,batch_desired_output = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={desired_input:batch_desired_input, \
    desired_output:batch_desired_output})
    
    print (i,sess.run(accuracy,feed_dict={desired_input:mnist.test.images, \
    desired_output:mnist.test.labels}))