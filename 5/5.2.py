#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  # 输入节点
OUTPUT_NODE = 10  # 输出节点
LAYER1_NODE = 500  # 隐藏层数

BATCH_SIZE = 100  # 每次batch打包的样本个数，一个训练中的数量数据个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8 # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARAZTION_RATE = 0.0001 # 正则化项在损失函数中的系数
TRAINING_STEPS = 5000 # 训练轮数滑动平均衰减率

# 辅助函数，输入参数，计算前向传播结果
# 定义 relu 激活函数的三层全连接，加入隐藏层实现多层结构
# relu激活函数去线性化
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        # 计算隐藏层前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层前向传播结果
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型的过程
def train(mnist):
    x = tf.placeholder (tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder (tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable (tf.truncated_normal ([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable (tf.constant (0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable (tf.truncated_normal ([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable (tf.constant (0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference (x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关的滑动平均类
    # 记录存储轮数变量
    global_step = tf.Variable (0, trainable=False)
    # 给定滑动平均衰减率和训练轮数，初始化滑动平均率
    variable_averages = tf.train.ExponentialMovingAverage (MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均
    variables_averages_op = variable_averages.apply (tf.trainable_variables ())
    # 计算使用滑动平均之后的前向传播结果
    average_y = inference (x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits (logits=y, labels=tf.argmax (y_, 1))
    cross_entropy_mean = tf.reduce_mean (cross_entropy)

    # L2正则化的损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer (REGULARAZTION_RATE)
    # 计算模型的正则化损失
    regularaztion = regularizer (weights1) + regularizer (weights2)
    #交叉熵+正则化损失
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay (
        LEARNING_RATE_BASE, # 基础学习率
        global_step, # 学习率在这个基础递减
        mnist.train.num_examples / BATCH_SIZE, # 过完所有训练数据需要迭代次数
        LEARNING_RATE_DECAY, # 学习率衰减速度
        staircase=True)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer (learning_rate).minimize (loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies ([train_step, variables_averages_op]):
        train_op = tf.no_op (name='train')

    # 计算正确率
    # 判断两个张量的每一维是否相等
    correct_prediction = tf.equal (tf.argmax (average_y, 1), tf.argmax (y_, 1))
    # 正确率
    accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session () as sess:
        tf.global_variables_initializer ().run ()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range (TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run (accuracy, feed_dict=validate_feed)
                print ("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs, ys = mnist.train.next_batch (BATCH_SIZE)
            sess.run (train_op, feed_dict={x: xs, y_: ys})
        # 训练结束后，测试测试数据检测的正确率
        test_acc = sess.run (accuracy, feed_dict=test_feed)
        print (("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))

# 主程序
def main(argv=None):
        mnist = input_data.read_data_sets ("./MNIST_data", one_hot=True)
        train (mnist)

if __name__ == '__main__':
        main()