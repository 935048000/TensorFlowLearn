# coding=utf-8
import tensorflow as tf


v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session () as sess:
    saver.restore (sess, "./Saved_model/model.ckpt")
    print(sess.run (result))


# 加载持久化的图，无需重复定义图上的运算
saver = tf.train.import_meta_graph("./Saved_model/model.ckpt")
with tf.Session() as sess:
    saver.restore(sess, "./Saved_model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

