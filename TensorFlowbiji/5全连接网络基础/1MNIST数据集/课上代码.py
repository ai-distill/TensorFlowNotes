from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/', one_hot=True) # 检查本地该路径是否有数据集，如果有就不下载，如果没有就下载
print(mnist.train.num_examples) # 打印训练集的样本数
print(mnist.validation.num_examples) # 打印验证集的样本数
print(mnist.test.num_examples)  # 打印测试集的样本数

# 返回标签和数据
print(mnist.train.labels[0])    # 查看训练集中指定编号的标签
print(mnist.train.images[0])    # 查看指定位置图的784个像素点

# 取一小撮数据，准备喂入神经网络
BATCH_SIZE = 200    # 定义一小撮是多少
xs, ys = mnist.train.next_batch(BATCH_SIZE)
print("xs shape : ", xs.shape)
print("ys shape : ", ys.shape) # 每个标签有10个元素，是输出的分类
