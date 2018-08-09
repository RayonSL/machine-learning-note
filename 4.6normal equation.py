'''
梯度下降与正规方程的比较：
梯度下降：需要选择学习率α，需要多次迭代，当特征数量n大时也能较好适用，适用于各种类型的模型
正规方程：不需要选择学习率α，一次计算得出，需要计算(XTX)−1，如果特征数量n较大则运算代价大，因为矩阵逆的计算时间复杂度为O(n3)，通常来说当n小于10000 时还是可以接受的，只适用于线性模型，不适合逻辑回归模型等其他模型
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path =  'D:\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

cols = data.shape[1]  # 列数
X = data.iloc[:,0:cols-1]  # 取前cols-1列，即输入向量
y = data.iloc[:,cols-1:cols] # 取最后一列，即目标向量

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0,0])

# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X)
    return theta
final_theta2=normalEqn(X, y)#感觉和批量梯度下降的theta的值有点差距

