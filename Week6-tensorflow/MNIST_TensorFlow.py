
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#导入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
#初始化参数
W = tf.Variable(tf.zeros([mnist.train.images.shape[1],10])) #W初始化为0
b = tf.Variable(tf.zeros([10])) #b初始化为0
#建立模型
x = tf.placeholder(tf.float32, [None, mnist.train.images.shape[1]]) 
y = tf.placeholder(tf.float32, [None, 10]) #建立训练集占位符

y_hat = tf.nn.softmax(tf.matmul(x,W) + b) #softmax激活
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1])) #成本函数 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost) #梯度下降，最小化成本
sess = tf.InteractiveSession() #创建session
tf.global_variables_initializer().run() #初始化变量（声明了变量，就必须初始化才能用） 

#迭代运算
costs = []

for epoch in range(1000): 
    batch_xs, batch_ys = mnist.train.next_batch(100) #每次使用100个小批量数据
    sess.run([train_step, cost], feed_dict = {x: batch_xs, y: batch_ys}) #进行训练

#计算精确度
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

