import tensorflow as tf
import reader

DATA_PATH = "../datasets/PTB/data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print (len(train_data))
print (train_data[:100])


# ptb_producer返回的为一个二维的tuple数据。
result = reader.ptb_producer(train_data, 4, 5)

# 通过队列依次读取batch。
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        x, y = sess.run(result)
        print ("X%d: "%i, x)
        print ("Y%d: "%i, y)
    coord.request_stop()
    coord.join(threads)