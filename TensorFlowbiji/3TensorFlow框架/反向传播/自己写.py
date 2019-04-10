# 0导入模块，准备数据集合
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

# 随机数生成器
rng = np.random.RandomState(SEED)
X = rng.rand(32, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]


# 1定义输入，输出以及具体的参数
# 预测值y，与已知答案y_之间的差距
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal(shape=(2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal(shape=(3, 1), stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 2准备反向传播过程，定义损失函数以及反向传播方法
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 3训练模型
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出未经训练的参数的值
    print("w1:\n", sess.run(w1))
    print("w2L\n", sess.run(w2))
    print("\n")

    # 训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training steps, loss on all data is %g" % (i, total_loss))

    # 输出训练后参数的值
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")
