# 搭建两层简单神经网络，并且采用占位符喂入一组数据
import tensorflow as tf

# 定义输入数据和参数
x = tf.placeholder(tf.float32, shape=(None, 2))         #注意这里要用float
w1 = tf.random_normal(shape=(2,3), stddev=1, seed=1)
w2 = tf.random_normal(shape=(3,1), stddev=1, seed=1)

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话执行计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y is ", sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))
    print("w1\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
