import tensorflow as tf

a = tf.nn.relu(tf.matmul(x,w1)+biases1)
y = tf.nn.relu(tf.matmul(a,w2)+biases2)