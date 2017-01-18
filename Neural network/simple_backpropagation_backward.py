import tensorflow as tf
import numpy as np

Number_of_neurons_i=3
Number_of_neurons_k=4
Number_of_neurons_j=5
Number_of_sets=300

def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
def bias_variable(shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)


#initial weights
w_ki=weight_variable([Number_of_neurons_i,Number_of_neurons_k])
w_jk=weight_variable([Number_of_neurons_k,Number_of_neurons_j])
b_k=bias_variable([Number_of_neurons_k])
b_j=bias_variable(([Number_of_neurons_j]))

#design your model
xs = tf.placeholder(tf.float32, [None, 3]) 
z_k=tf.matmul(xs,w_ki)+b_k
a_k=tf.sigmoid(z_k)
z_j=tf.matmul(a_k,w_jk)+b_j
a_j=tf.sigmoid(z_j)

sess=tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

input_data=np.random.rand(Number_of_sets,Number_of_neurons_i)
output_data=np.random.rand(Number_of_sets,Number_of_neurons_j)

print "Before training: wights of w_ki\n",sess.run(w_ki)

#define cost function
os = tf.placeholder("float", [None,Number_of_neurons_j])
cost = tf.reduce_mean(tf.reduce_sum(tf.square(os - a_j),reduction_indices=1))

train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cost)
sess.run(train_step,feed_dict={xs:input_data,os:output_data})


print "After training: wights of w_ki\n",sess.run(w_ki)

#There are so many optimizers you can chose
'''
AdadeltaOptimizer
AdagradDAOptimizer
MomentumOptimizer
AdamOptimizer
FtrlOptimizer
ProximalGradientDescentOptimizer
ProximalAdagradOptimizer
RMSPropOptimizer
'''