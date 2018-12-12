import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import spectral
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# 导入数据库
input_mat2 = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data2_train.mat')['data2_train']
input_mat3 = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data3_train.mat')['data3_train']
input_mat5 = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data5_train.mat')['data5_train']
input_mat6 = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data6_train.mat')['data6_train']
input_mat8 = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data8_train.mat')['data8_train']
input_mat10 = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data10_train.mat')['data10_train']
input_mat11 = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data11_train.mat')['data11_train']
input_mat12 = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data12_train.mat')['data12_train']
input_mat14 = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data14_train.mat')['data14_train']

#合并样本集
merge = np.concatenate([input_mat2,input_mat3,input_mat5,input_mat6,input_mat8,input_mat10,input_mat11,input_mat12,input_mat14],axis = 0)
# print(merge.shape) # (6924,200)

target = np.zeros(shape=(merge.shape[0],1))

#合并标签
y2 = input_mat2.shape[0]
for i in range(y2):
    target[i]=2
y3 = input_mat3.shape[0] + y2
for i in range(y2,y3):
    target[i]=3
y5 = input_mat5.shape[0] + y3
for i in range(y3,y5):
    target[i]=5
y6 = input_mat6.shape[0] + y5
for i in range(y5,y6):
    target[i]=6
y8 = input_mat8.shape[0] + y6
for i in range(y6,y8):
    target[i]=8
y10 = input_mat10.shape[0] + y8
for i in range(y8,y10):
    target[i]=10
y11 = input_mat11.shape[0] + y10
for i in range(y10,y11):
    target[i]=11
y12 = input_mat12.shape[0] + y11
for i in range(y11,y12):
    target[i]=12
y14 = input_mat14.shape[0] + y12
for i in range(y12,y14):
    target[i]=14

#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(merge, target,test_size=0.2)

# 预处理规范化并降维
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# PCA降维
def pca(dataMat, topNfeat=999999):

    # 1.对所有样本进行中心化（所有样本属性减去属性的平均值）
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals

    # 2.计算样本的协方差矩阵 XXT
    covmat = np.cov(meanRemoved, rowvar=0)
    print(covmat)

    # 3.对协方差矩阵做特征值分解，求得其特征值和特征向量，并将特征值从大到小排序，筛选出前topNfeat个
    eigVals, eigVects = np.linalg.eig(np.mat(covmat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]    # 取前topNfeat大的特征值的索引
    redEigVects = eigVects[:, eigValInd]        # 取前topNfeat大的特征值所对应的特征向量

    # 4.将数据转换到新的低维空间中
    lowDDataMat = meanRemoved * redEigVects     # 降维之后的数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # 重构数据，可在原数据维度下进行对比查看
    return np.array(lowDDataMat), np.array(reconMat)

pca = PCA()
x_train = pca.fit_transform(x_train)
y_train = pca.fit_transform(y_train)

# 训练模型
clf = SVC(kernel = 'rbf',gamma=0.125,C=15)
y = y_train.ravel()
y_train = np.array(y).astype(int)
clf.fit(x_train,y_train)
pre = clf.predict(x_test)
# print(pre)
# print(clf.score(x_train,y_train)) # 0.9992076
accuracy = metrics.accuracy_score(y_test,pre)
# print(accuracy) # 0.88375

#加载测试集
test = sio.loadmat('/Users/ZhangHao 1/Documents/GitHub/GitHubRepository/18b-zhanghao-2016-667/ClassificationSchoolwork/dataset/data_test_final.mat')['data_test_final']
test = np.array(test, dtype=np.float64)
test = StandardScaler().fit_transform(test)
x_test = scaler.transform(test)
# print(x_test)

#生成测试集标签导出csv
y_pre = clf.predict(x_test)
y_pre = pd.DataFrame(np.c_[np.arange(y_pre.shape[0]),y_pre],columns=['id',y])
# print(y_pre)
y_pre.to_csv('y_pre.csv',index=False)