#############################多变量线性回归##########################
#练习1还包括一个房屋价格数据集，其中有2个变量（房子的大小，卧室的数量）和目标（房子的价格）。 我们使用我们已经应用的技术来分析数据集。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path =  'D:\ex2data2.txt'
data2 = pd.read_csv(path, names=['Size', 'Bedrooms', 'Price'])
data2.head()

#对于此任务，我们添加了另一个预处理步骤 - 特征归一化。 这个对于pandas来说很简单
data2 = (data2 - data2.mean()) / data2.std()
data2.head()

#现在我们重复第1部分的预处理步骤，并对新数据集运行线性回归程序。
# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

#计算代价函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) -  y), 2)
    return np.sum(inner) / (2 * len(X))

#计算线性回归
def gradientDescent(X, y, theta, alpha, epoch):
    """reuturn theta, cost"""
    temp = np.matrix(np.zeros(theta.shape))  # 初始化一个 θ 临时矩阵(1, 2)
    parameters = int(theta.flatten().shape[1])  # 参数 θ的数量
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m

    for i in range(epoch):
        # 利用向量化一步求解
        temp =theta - (alpha / m) * (X * theta.T - y).T * X

# 以下是不用Vectorization求解梯度下降
#         error = (X * theta.T) - y  # (97, 1)

#         for j in range(parameters):
#             term = np.multiply(error, X[:,j])  # (97, 1)
#             temp[0,j] = theta[0,j] - ((alpha / m) * np.sum(term))  # (1,1)

        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

#初始化一些附加变量 - 学习速率α和要执行的迭代次数。
alpha = 0.01
epoch = 1000

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, epoch)

# get the cost (error) of the model
computeCost(X2, y2, g2), g2

#查看训练进程
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(epoch), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
