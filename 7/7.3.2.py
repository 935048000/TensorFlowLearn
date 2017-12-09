import tensorflow as tf

# 生成文件存储样例数据。
# 创建TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

num_shards = 2
instances_per_shard = 2 # 每个文件有多少数据
for i in range(num_shards):
    # 多个文件区分：0000n-of-0000m ，n=文件编号，m=数据被存在多少个文件中
    filename = ('./Records/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    # 将Example结构写入TFRecord文件。
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shard):
    # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()


#  读取文件。
files = tf.train.match_filenames_once("Records/data.tfrecords-*")
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
      serialized_example,
      features={
          'i': tf.FixedLenFeature([], tf.int64),
          'j': tf.FixedLenFeature([], tf.int64),
      })
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    print (sess.run(files))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(6):
        print (sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)

# 结果：生成两个文件，每个文件存两个样例





