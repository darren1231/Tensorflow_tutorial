# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:27:05 2017

@author: darren
"""

import tensorflow as tf  
  
#our NN's output  
net_output=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])  
#step1:do softmax  
net_soft_output=tf.nn.softmax(net_output)  
#true label  
desired_output=tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])  
#step2:do cross_entropy  
cross_entropy = tf.reduce_mean(-tf.reduce_sum(desired_output*tf.log(net_soft_output),reduction_indices=[1]))
#do cross_entropy just one step  
cross_entropy2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits \
(labels=desired_output, logits=net_output ))#dont forget tf.reduce_sum()!!  

with tf.Session() as sess:  
    softmax=sess.run(net_output)  
    c_e = sess.run(cross_entropy)  
    c_e2 = sess.run(cross_entropy2)  
    print("step1:softmax result=")  
    print(softmax)  
    print("step2:cross_entropy result=")  
    print(c_e)  
    print("Function(softmax_cross_entropy_with_logits) result=")  
    print(c_e2)  
    
    
