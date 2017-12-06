# coding=utf-8
import tensorflow as tf

# # 使用滑动平均
# v = tf.Variable (0, dtype=tf.float32, name="v")
# for variables in tf.global_variables ():
#     print(variables.name)
#
# ema = tf.train.ExponentialMovingAverage (0.99)
# maintain_averages_op = ema.apply (tf.global_variables ())
# for variables in tf.global_variables ():
#     print(variables.name)



# 保存滑动平均模型。
# saver = tf.train.Saver ()
# with tf.Session () as sess:
#     init_op = tf.global_variables_initializer ()
#     sess.run (init_op)
#
#     sess.run (tf.assign (v, 10))
#     sess.run (maintain_averages_op)
#     # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
#     saver.save (sess, "./Saved_model/model2.ckpt")
#     print(sess.run ([v, ema.average (v)]))


# v = tf.Variable(0, dtype=tf.float32, name="v")
# # 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
# saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
# with tf.Session() as sess:
#     saver.restore(sess, "./Saved_model/model2.ckpt")
#     print(sess.run(v))


# # variables_to_restore() 使用案例
# v = tf.Variable(0, dtype=tf.float32, name="v")
# ema = tf.train.ExponentialMovingAverage(0.99)
# print (ema.variables_to_restore())
#
# saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
# with tf.Session() as sess:
#     saver.restore(sess, "./Saved_model/model2.ckpt")
#     print (sess.run(v))


# pb文件的保存方法
import tensorflow as tf
from tensorflow.python.framework import graph_util
#
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
# result = v1 + v2
#
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     graph_def = tf.get_default_graph().as_graph_def()
#     output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
#     with tf.gfile.GFile("./Saved_model/combined_model.pb", "wb") as f:
#            f.write(output_graph_def.SerializeToString())


# # 加载pb文件
from tensorflow.python.platform import gfile
with tf.Session () as sess:
    model_filename = "./Saved_model/combined_model.pb"

    with gfile.FastGFile (model_filename, 'rb') as f:
        graph_def = tf.GraphDef ()
        graph_def.ParseFromString (f.read ())

    result = tf.import_graph_def (graph_def, return_elements=["add:0"])
    print(sess.run (result))
