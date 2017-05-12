# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:46:28 2017

@author: darren
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initail = tf.constant(0.1,shape=shape)
    return tf.Variable(initail)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    

#network weights
w_conv1= weight_variable([5,5,1,32])
b_conv1= bias_variable([32])

w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])


desired_input = tf.placeholder(tf.float32,[None,784])
desired_output = tf.placeholder(tf.float32,[None,10])

x_image = tf.reshape(desired_input,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
print ("h_conv1:",h_conv1.get_shape())
h_pool1 = max_pool_2x2(h_conv1)
print ("h_pool1:",h_pool1.get_shape())
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
print ("h_conv2:",h_conv2.get_shape())
h_pool2 = max_pool_2x2(h_conv2)
print ("h_pool2:",h_pool2.get_shape())
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
print ("h_pool2_flat:",h_pool2_flat.get_shape())
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
print ("h_fc1:",h_fc1.get_shape())
net_output = tf.nn.softmax(tf.matmul(h_fc1,w_fc2)+b_fc2)
print ("net_output:",net_output.get_shape())

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits \
#(labels=desired_output, logits=net_output))
cross_entropy =tf.reduce_mean(-tf.reduce_sum(desired_output*tf.log(net_output) \
,reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.arg_max(desired_output,1),tf.arg_max(net_output,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess=tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in range(200):
    print (str(i)+" ",end='') 
    batch_desired_input,batch_desired_output= mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={desired_input:batch_desired_input, \
    desired_output:batch_desired_output})
    if i%100==0:
        
        print (i,sess.run(accuracy,feed_dict={desired_input:mnist.test.images, \
    desired_output:mnist.test.labels}))