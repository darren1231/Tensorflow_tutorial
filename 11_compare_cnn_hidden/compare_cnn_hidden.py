# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:20:33 2017

@author: darren
"""

import tensorflow as tf
import sys
import os

#versioning, urllib named differently for dif python versions
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
    
# tsv is the file which contain the label of each picture.
# png is the file which put all of the small picture together.
# obtain the location where you run the code   
os_location=os.getcwd()
LOGDIR = os_location+'/embedding_data/'
GITHUB_URL ='https://raw.githubusercontent.com/darren1231/Tensorflow_tutorial/master/10_hidden_mnist_embedding/'
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)
urlretrieve(GITHUB_URL + 'labels_1024.tsv', LOGDIR + 'labels_1024.tsv')
urlretrieve(GITHUB_URL + 'sprite_1024.png', LOGDIR + 'sprite_1024.png')

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
    
def cnn_model():
    
    desired_input = tf.placeholder(tf.float32,[None,784])
    desired_output = tf.placeholder(tf.float32,[None,10])    
    x_image = tf.reshape(desired_input,[-1,28,28,1])
    #network weights
    with tf.name_scope("cnn_net"):
        
        with tf.name_scope("w_conv_1"):
            w_conv1= weight_variable([5,5,1,32])
            b_conv1= bias_variable([32])
            tf.summary.histogram("w_conv1",w_conv1)
            tf.summary.histogram("b_conv1",b_conv1)
        with tf.name_scope("w_conv_2"):
            w_conv2 = weight_variable([5,5,32,64])
            b_conv2 = bias_variable([64])
            tf.summary.histogram("w_conv2",w_conv2)
            tf.summary.histogram("b_conv2",b_conv2)
        
        with tf.name_scope("fc1"):
            w_fc1 = weight_variable([7*7*64,1024])
            b_fc1 = bias_variable([1024])
            tf.summary.histogram("w_fc1",w_fc1)
            tf.summary.histogram("b_fc1",b_fc1)
            
        with tf.name_scope("fc2"):
            w_fc2 = weight_variable([1024,10])
            b_fc2 = bias_variable([10])
            tf.summary.histogram("w_fc2",w_fc2)
            tf.summary.histogram("b_fc2",b_fc2)       
        
        
        with tf.name_scope("conv1"):
            h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)    
            h_pool1 = max_pool_2x2(h_conv1)    
        with tf.name_scope("conv2"):
            h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)    
            h_pool2 = max_pool_2x2(h_conv2)    
        
        with tf.name_scope("fully_connected"):
            h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])    
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)    
            net_output = tf.nn.softmax(tf.matmul(h_fc1,w_fc2)+b_fc2)
    
    embedding_input = net_output
    embedding_size = 10
    
    return desired_input,desired_output,net_output,embedding_input,embedding_size
    
def hidden_model():
    desired_input = tf.placeholder(tf.float32,[None,784])
    desired_output = tf.placeholder(tf.float32,[None,10])
    
    with tf.name_scope("784_10_network"):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.zeros([784,10]))
            tf.summary.histogram("weights",weights)
            
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([10]))
            tf.summary.histogram("biases",biases)    
            
        with tf.name_scope("net_output"):
            net_output=tf.nn.softmax(tf.matmul(desired_input,weights)+biases)
            embedding_input = net_output
            embedding_size = 10
            tf.summary.histogram("net_output",net_output)
            
    return desired_input,desired_output,net_output,embedding_input,embedding_size

def train_model(use_cnn,learning_rate):
    tf.reset_default_graph()
    
    if use_cnn:
        desired_input,desired_output,net_output,embedding_input,embedding_size=cnn_model()
#        embedding_name="hidden_1024"
        experiment_name="cnn_net"+str(learning_rate)+"/"
    else:
        desired_input,desired_output,net_output,embedding_input,embedding_size=hidden_model()
#        embedding_name="output_10"
        experiment_name="hidden_net"+str(learning_rate)+"/"
    
    with tf.name_scope("train"):
        with tf.name_scope("loss"):
            loss_cross_entrop =tf.reduce_mean(-tf.reduce_sum(desired_output* \
            tf.log(net_output),reduction_indices=[1]))
            tf.summary.scalar("loss",loss_cross_entrop)
        with tf.name_scope("train_step"):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_cross_entrop)
        
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.arg_max(desired_output,1),tf.arg_max(net_output,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            tf.summary.scalar("accuracy",accuracy)
            
    # If you want to see the picture in tensorboard, you can use summary.image
    # function. The max number of pictures is 3 and present in gray scale(1).
    # If you want to use RGB instead, you should change 1 to 3.
    x_image = tf.reshape(desired_input, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    
    #tensorboard merged all
    merged= tf.summary.merge_all()
    
    #intiialize embedding matrix as 0s
    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="embedding")
    #give it calculated embedding
    assignment = embedding.assign(embedding_input)
    
    saver = tf.train.Saver()
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    writer = tf.summary.FileWriter(LOGDIR+experiment_name,sess.graph)
    writer.add_graph(sess.graph)
    
    ## Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    ## You can add multiple embeddings. Here we add only one.
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = LOGDIR + 'sprite_1024.png'
    embedding_config.metadata_path = LOGDIR + 'labels_1024.tsv'
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    
    
    
    for i in range(5001):
        
        batch_desired_input,batch_desired_output = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={desired_input:batch_desired_input, \
        desired_output:batch_desired_output})
        
        if i%200==0:
            print ("step:",i," accuracy:",sess.run(accuracy,feed_dict={desired_input:mnist.test.images[:1024], \
            desired_output:mnist.test.labels[:1024]}))
            
        summary_data=sess.run(merged,feed_dict={desired_input:batch_desired_input, \
        desired_output:batch_desired_output})
        writer.add_summary(summary_data,i)
        
        if i%1000==0:
            
            sess.run(assignment, feed_dict={desired_input: mnist.test.images[:1024], desired_output: mnist.test.labels[:1024]})
            #save checkpoints
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)

def main():
    for learning_rate in [1e-3,1e-4,1e-5]:
        for use_cnn in [False,True]:
            print ("use_cnn:",use_cnn," learning_rate:",learning_rate)
            train_model(use_cnn,learning_rate)
            
    
if __name__ == '__main__':
    main()