import tensorflow as tf
import numpy as np

def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
def bias_variable(shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
xs = tf.placeholder(tf.float32, [None, 3])  

w_ki=weight_variable([3,4])
w_jk=weight_variable([4,5])
b_k=bias_variable([4])
b_j=bias_variable(([5]))

z_k=tf.matmul(xs,w_ki)+b_k
a_k=tf.sigmoid(z_k)
z_j=tf.matmul(a_k,w_jk)+b_j
a_j=tf.sigmoid(z_j)

input_data=np.random.rand(2,3)

sess=tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

out=sess.run(a_j,feed_dict={xs:input_data})

print out