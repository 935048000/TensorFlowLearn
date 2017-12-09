import tensorflow as tf

## FIFOQueue 函数的使用
# 创建队列，并操作里面的元素。先进先出
q = tf.FIFOQueue (2, "int32")
init = q.enqueue_many (([0, 10],))
x = q.dequeue ()
y = x + 1
q_inc = q.enqueue ([y])
with tf.Session () as sess:
    init.run ()
    for _ in range (5):
        v, _ = sess.run ([x, q_inc])
        print (v)
print ("---------------------------------\n")



## Coordinator函数的使用
import numpy as np
import threading
import time
#  这个程序每隔1秒判断是否需要停止并打印自己的ID。
def MyLoop(coord, worker_id):
    while not coord.should_stop ():
        # 随机停止所有进程
        if np.random.rand () < 0.1:
            print ("Stoping from id: %d" % worker_id)
            # 通知线程停止
            coord.request_stop ()
        else:
            # 打印ID
            print ("Working on id: %d" % worker_id)
        time.sleep (1)

# 协同多线程
coord = tf.train.Coordinator ()
# 创建5个线程
threads = [threading.Thread (target=MyLoop, args=(coord, i,)) for i in range (5)]
# 启动所有线程
for t in threads: t.start ()
# 等待线程退出
coord.join (threads)

print ("---------------------------------\n")


## QueueRunner + Coordinator 函数的使用
import tensorflow as tf
# 定义队列及其操作
# 声明队列
queue = tf.FIFOQueue(100,"float")
# 入队
enqueue_op = queue.enqueue([tf.random_normal([1])])
# 多线程入队
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
# qr加入技计算图指定集合
tf.train.add_queue_runner(qr)
# 出队
out_tensor = queue.dequeue()
#  启动线程。
with tf.Session() as sess:
    # 协同
    coord = tf.train.Coordinator()
    # 起线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取队列中的值
    for _ in range(3): print (sess.run(out_tensor)[0])
    # 停止所有进程
    coord.request_stop()
    coord.join(threads)