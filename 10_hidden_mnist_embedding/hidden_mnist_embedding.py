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
GITHUB_URL ='https://raw.githubusercontent.com/darren1231/Tensorflow_tutorial/master/8_hidden_mnist_embedding/'
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)
urlretrieve(GITHUB_URL + 'labels_1024.tsv', LOGDIR + 'labels_1024.tsv')
urlretrieve(GITHUB_URL + 'sprite_1024.png', LOGDIR + 'sprite_1024.png')

tf.reset_default_graph()

desired_input = tf.placeholder(tf.float32,[None,784])
desired_output = tf.placeholder(tf.float32,[None,10])

# If you want to see the picture in tensorboard, you can use summary.image
# function. The max number of pictures is 3 and present in gray scale(1).
# If you want to use RGB instead, you should change 1 to 3.
x_image = tf.reshape(desired_input, [-1, 28, 28, 1])
tf.summary.image('input', x_image, 3)

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

with tf.name_scope("train"):
    with tf.name_scope("loss"):
        loss_cross_entrop =tf.reduce_mean(-tf.reduce_sum(desired_output* \
        tf.log(net_output),reduction_indices=[1]))
        tf.summary.scalar("loss",loss_cross_entrop)
    with tf.name_scope("train_step"):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_cross_entrop)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.arg_max(desired_output,1),tf.arg_max(net_output,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar("accuracy",accuracy)


#tensorboard merged all
merged= tf.summary.merge_all()

#intiialize embedding matrix as 0s
embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="out_put_10")
#give it calculated embedding
assignment = embedding.assign(embedding_input)

saver = tf.train.Saver()

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

writer = tf.summary.FileWriter(LOGDIR+"event_data/",sess.graph)
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



for i in range(2000):
    
    batch_desired_input,batch_desired_output = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={desired_input:batch_desired_input, \
    desired_output:batch_desired_output})
    
    print ("step:",i," accuracy:",sess.run(accuracy,feed_dict={desired_input:mnist.test.images, \
    desired_output:mnist.test.labels}))
    
    summary_data=sess.run(merged,feed_dict={desired_input:batch_desired_input, \
    desired_output:batch_desired_output})
    writer.add_summary(summary_data,i)
    
    if i%100==0:
        
        sess.run(assignment, feed_dict={desired_input: mnist.test.images[:1024], desired_output: mnist.test.labels[:1024]})
        #save checkpoints
        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)