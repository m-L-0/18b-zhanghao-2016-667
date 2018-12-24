#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import sklearn.metrics as sm

#导入数据集
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

data = X[:,[1,3]] #取两个维度

# 可视化输出图
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-lenght','petal-width','class']
dataset = pd.read_csv(url,names = names) #读取csv数据
print(dataset.describe()) #输出数据集
dataset.hist() #直方图
dataset.plot(x='sepal-length', y='sepal-width', kind='scatter') #散点图
plt.show()

iris_setosa = iris.data[:50]
iris_versicolor = iris.data[50:100]
iris_virginica = iris.data[100:150]

iris_setosa = np.hsplit(iris_setosa,4) #水平分割获取各特征集合，分割成4列
iris_versicolor = np.hsplit(iris_versicolor,4)
iris_virginica = np.hsplit(iris_virginica,4)

setosa = {'sepal_length':iris_setosa[0],'sepal_width':iris_setosa[1],'petal length':iris_setosa[2],'petal width':iris_setosa[3]}

versicolor = {'sepal_length':iris_versicolor[0],'sepal_width':iris_versicolor[1],'petal length':iris_versicolor[2],'petal width':iris_versicolor[3]}

virginica = {'sepal_length':iris_virginica[0],'sepal_width':iris_virginica[1],'petal length':iris_virginica[2],'petal width':iris_virginica[3]}

size = 5 #散点大小
setosa_color = 'r'
versicolor_color = 'g'
virginica_color = 'b'

sepal_length_ticks = np.arange(4,8,step=0.5)
sepal_width_ticks = np.arange(2,5,step=0.5)
petal_length_ticks = np.arange(1,7,step=1)
petal_width_ticks = np.arange(0,2.5,step=0.5)

ticks = [sepal_length_ticks,sepal_width_ticks,petal_length_ticks,petal_width_ticks]
label_text = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']

print(ticks)

plt.figure(figsize=(12,12)) #画布大小
plt.suptitle("Iris Set (red=setosa,green=versicolor,blue=virginica")

for i in range(0,4):
    for j in range(0,4):
        plt.subplot(4,4,i * 4 + j + 1) #创建子画布
        if i == j:
            print(i*4+j+1) #序列号
            plt.xticks([])
            plt.yticks([])
            plt.text(0.1,0.4,label_text[i],size = 18)
        else:
            plt.scatter(iris_setosa[j],iris_setosa[i],c = setosa_color,s = size)
            plt.scatter(iris_versicolor[j],iris_versicolor[i],c = versicolor_color,s = size)
            plt.scatter(iris_virginica[j],iris_virginica[i],c = virginica_color,s = size)
            plt.xticks(ticks[j])
            plt.yticks(ticks[i])

plt.show()

#计算欧式距离
def distance(p1,p2):
    tmp = np.sum((p1-p2)**2)
    return np.sqrt(tmp)

# 邻接矩阵
def get_dis_matrix(X): 
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)): 
            S[i][j] = 1.0 * distance(X[i], X[j]) 
            S[j][i] = S[i][j] 
    return S
S = get_dis_matrix(data)

def getA(S, k, sigma = 1):
    N = len(S)
    A = np.zeros((N, N))
    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key = lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)]
        for j in neighbours_id:
            A[i][j] = np.exp(-S[i][j]/ 2/ sigma/ sigma)
            A[j][i] = A[i][j]
    return A
A = getA(S, 10)

# 度矩阵
def getW(data,k):
    points_num = len(iris.data)
    dis_matrix = np.zeros((points_num,points_num))
    W = np.zeros((points_num,points_num))
    for i in range(points_num):
        for j in range(i+1,points_num):
            dis_matrix[i][j] = dis_matrix[j][i] = distance(data[i],data[j])
    for idx,each in enumerate(dis_matrix):
        index_array = np.argsort(each)
        W[idx][index_array[1:k+1]] = 1
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2
    return W
W = getW(data,150)

def getD(W):
    points_num = len(W)
    D = np.diag(np.zeros(points_num))
    for i in range(points_num):
        D[i][i] = sum(W[i])
    return D
D = getD(W)
print(D)

#拉普拉斯矩阵
def getL(D,A):
    L = np.zeros(150*150)
    L = L.reshape(150*150)
    L = D-A
    return L
L = getL(D,A)
print(L)

# KMeans 聚类

# 将输入存储为datarame并设置列名
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

# 创建模型
model = KMeans(n_clusters=3)
model.fit(x)

# 标签
label_pred = model.labels_
predY = np.choose(model.labels_, [1,0,2]).astype(np.int64)
print(label_pred)
print(predY)

# 重新输出图
plt.figure(figsize=(14,7))
colormap = np.array(['red','green','blue'])
plt.subplot(1,2,1)
plt.scatter(x.Sepal_Length, x.Sepal_Width, c=colormap[y.Targets],s=40)
plt.title('Real')
plt.subplot(1,2,2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('KMeans')
plt.show()

# 混淆矩阵
matrix = sm.confusion_matrix(y,predY)
print(matrix)

# 准确率 acc
acc = sm.accuracy_score(y,predY)
print(acc)