#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import numpy as np
from sklearn import datasets
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 按照8：2划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1,shuffle=True)

# # 预处理
# std = StandardScaler()
# X_train = std.fit_transform(X_train)
# X_test = std.fit_transform(X_test)

# 设置占位符
train = tf.placeholder("float", [None, 4])
test = tf.placeholder("float", [4])

# L1 距离
distance = tf.reduce_sum(tf.abs(tf.add(train, tf.negative(test))), reduction_indices=1)

# 定义准确率
acc = 0

# 预测最小距离，调整K
pred = tf.arg_min(distance, 0)

# 初始化变量
init = tf.initialize_all_variables()

# 算法
with tf.Session() as sess:
    sess.run(init)
    # 遍历测试集
    for i in range(len(X_test)):
        # 得到最近邻
        nn_index = sess.run(pred, feed_dict={train:X_train, test: X_test[i]})
        print("最近邻", nn_index)
        # 得到最近邻标签并比较
        print("Sample", i, " - Prediction:", y_train[nn_index], " / True Class:", y_test[i])
        # 计算准确率
        if y_train[nn_index] == y_test[i]:
            acc += 1. / len(X_test)
print("Accuracy:", acc)
