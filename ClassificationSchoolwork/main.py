#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# 张浩-2016011667-ClassificationSchoolwork
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 导入数据库
input_mat2 = sio.loadmat('data2_train.mat')['data2_train']
input_mat3 = sio.loadmat('data3_train.mat')['data3_train']
input_mat5 = sio.loadmat('data5_train.mat')['data5_train']
input_mat6 = sio.loadmat('data6_train.mat')['data6_train']
input_mat8 = sio.loadmat('data8_train.mat')['data8_train']
input_mat10 = sio.loadmat('data10_train.mat')['data10_train']
input_mat11 = sio.loadmat('data11_train.mat')['data11_train']
input_mat12 = sio.loadmat('data12_train.mat')['data12_train']
input_mat14 = sio.loadmat('data14_train.mat')['data14_train']

#合并样本集
merge = np.concatenate([input_mat2,input_mat3,input_mat5,input_mat6,input_mat8,input_mat10,input_mat11,input_mat12,input_mat14],axis = 0)
print(merge.shape) # (6924,200)
data = merge.reshape((-1,200))
print(data)

#合并标签
label = np.zeros(shape=(merge.shape[0],1))
label = np.array(label,dtype=np.int32)

y2 = input_mat2.shape[0]
for i in range(y2):
    label[i]=2
y3 = input_mat3.shape[0] + y2
for i in range(y2,y3):
    label[i]=3
y5 = input_mat5.shape[0] + y3
for i in range(y3,y5):
    label[i]=5
y6 = input_mat6.shape[0] + y5
for i in range(y5,y6):
    label[i]=6
y8 = input_mat8.shape[0] + y6
for i in range(y6,y8):
    label[i]=8
y10 = input_mat10.shape[0] + y8
for i in range(y8,y10):
    label[i]=10
y11 = input_mat11.shape[0] + y10
for i in range(y10,y11):
    label[i]=11
y12 = input_mat12.shape[0] + y11
for i in range(y11,y12):
    label[i]=12
y14 = input_mat14.shape[0] + y12
for i in range(y12,y14):
    label[i]=14

print(label)
print(label.shape)

#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data, label,test_size=0.2,random_state = 1)

# 降维
pca = PCA(n_components='mle')
pca.fit_transform(x_train)
pca.transform(x_test)

print(pca.explained_variance_ratio_.shape) #180

# 构建模型
y = y_train.ravel()
y_train = np.array(y).astype(int)
#调参
a = [3,4,5,6,7]
for i in a:
    svc = SVC(kernel='poly', gamma='auto', degree=i)
    svc.fit(x_train,y_train)
    print(svc.score(x_test,y_test))

clf = SVC(kernel='poly', gamma='auto', degree=5 )
clf.fit(x_train,y_train)

# 导入测试集
test = sio.loadmat('data_test_final.mat')['data_test_final']
final = clf.predict(test)
final = np.array(final)
print(final)
data_final = pd.DataFrame(final)
data_final.to_csv('final.csv')