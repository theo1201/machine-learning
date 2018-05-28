 # 手写数据集的输入

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, 784], name='x-input')
	y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔类型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
# 将bool类型的预测结果转换为数值类型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(init)
	#     将文件的结构写入logs文件中
	writer = tf.summary.FileWriter('logs/', sess.graph)
	for epoch in range(1):
		for batch in range(n_batch):
			#             图片的数据保存在xs中，图片的标签保存在ys中
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

		acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print("Iter" + str(epoch) + ",Tesing accuracy" + str(acc))

