import tensorflow as tf

# Create a variable.
w = tf.Variable(10.0, name="w")
#w = tf.constant(10.0, name="w")   #You can unmark this line to see what happen

# Assign a new value to the variable with `assign()` or a related method.
assign_function=w.assign(w + 1.0)

sess=tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

sess.run(assign_function)
print sess.run(w)

sess.run(assign_function)
print sess.run(w)