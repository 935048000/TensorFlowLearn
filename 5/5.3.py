# coding=utf-8
import tensorflow as tf


with tf. variable_scope("foo"):
    v = tf. get_variable("v", [1], initializer=tf. constant_initializer(1.0))

# with tf.variable_scope("foo"):
# v = tf.get_variable("v", [1])

with tf. variable_scope("foo", reuse=True):
    v1 = tf. get_variable("v", [1])

print (v == v1)

# with tf.variable_scope("bar", reuse=True):
#    v = tf.get_variable("v", [1])

with tf.variable_scope ("root"):
    # tf.get_variable_scope () 获取上下文中管理器的reuse
    print(tf.get_variable_scope ().reuse) # 最外层reuse为false


    with tf.variable_scope ("foo", reuse=True): # 新建一个嵌套的上下文管理器
        print(tf.get_variable_scope ().reuse)   # 指定reuse为true，且输出


        with tf.variable_scope ("bar"): # 新建一个嵌套的上下文管理器，不指定reuse
            print(tf.get_variable_scope ().reuse)

    print(tf.get_variable_scope ().reuse) # 退出reuse为true的上下文后为

print("\n\n")

v1 = tf.get_variable ("v", [1])
print(v1.name) # 输出 变量名：变量


with tf.variable_scope ("foo", reuse=True):
    v2 = tf.get_variable ("v", [1])
print(v2.name)


with tf.variable_scope ("foo"):
    with tf.variable_scope ("bar"):
        v3 = tf.get_variable ("v", [1])
        print(v3.name)


v4 = tf.get_variable ("v1", [1])
print(v4.name)

print()

with tf.variable_scope("",reuse=True):
    v5 = tf.get_variable("foo/bar/v", [1])
    print (v5 == v3)
    v6 = tf.get_variable("v1", [1])
    print (v6 == v4)



