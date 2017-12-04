#coding=utf-8
import tensorflow as tf
from numpy.random import RandomState

# 训练数据大小
batch_size = 8

# 神经网络参数
w1 = tf.Variable (tf.random_normal ([2, 3], stddev=1, seed=1))
w2 = tf.Variable (tf.random_normal ([3, 1], stddev=1, seed=1))

# 方便使用不大的batch大小
x = tf.placeholder (tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder (tf.float32, shape=(None, 1), name="y-input")

#定义前向传播
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# 计算交叉熵（损失函数）和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)

# 随机模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

Y = [[int(x1+x2 < 1)] for (x1,x2) in X]

# 创建运行tf的程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1:",sess.run(w1))
    print("w2:",sess.run(w2))
    print("\n")
# 输出训练前参数的值

    # 训练轮数
    STEPS = 6000
    for i in range(STEPS):
        # 每次取batch_size个样本训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)

        # 训练并更新参数
        sess.run(train_step,
                 feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(
                cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross entropy on all data is %g"%
                  (i,total_cross_entropy))
# 交叉熵越小，预测数据和真实数据差距越小
    print("w1:",sess.run(w1))
    print("w2:",sess.run(w2))