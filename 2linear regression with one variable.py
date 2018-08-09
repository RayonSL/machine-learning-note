##################################单变量线性回归##############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path =  'D:\ex1data1.txt'

# names添加列名，header用指定的行来作为标题，若原无标题且指定标题则设为None
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])#header ：指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0【第一行数据】，否则设置为None。  
data.head()

data.describe()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
plt.show()

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) -  y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]  # 列数
X = data.iloc[:,0:cols-1]  # 取前cols-1列，即输入向量
y = data.iloc[:,cols-1:cols] # 取最后一列，即目标向量

X.head()  # head()是观察前5行
y.head()  # head()是观察前5行

'''
注意：这里我使用的是matix而不是array，两者基本通用。

但是matrix的优势就是相对简单的运算符号，比如两个矩阵相乘，就是用符号*，但是array相乘不能这么用，得用方法.dot() 
array的优势就是不仅仅表示二维，还能表示3、4、5…维，而且在大部分Python程序里，array也是更常用的。

两者区别： 
1. 对应元素相乘：matrix可以用np.multiply(X2,X1)，array直接X1*X2 
2. 点乘：matrix直接X1*X2，array可以 X1@X2 或 X1.dot(X2) 或 np.dot(X1, X2)

代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
'''
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0,0])

np.array([[0,0]]).shape 
# (1, 2)

X.shape, theta.shape, y.shape
# ((97, 2), (1, 2), (97, 1))

computeCost(X, y, theta) # 32.072733877455676

X.shape, theta.shape, y.shape, X.shape[0]
# ((97, 2), (1, 2), (97, 1), 97)

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
#现在让我们运行梯度下降算法来将我们的参数θ适合于训练集。
final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)
#最后，我们可以使用我们拟合的参数计算训练模型的代价函数（误差）。
computeCost(X, y, final_theta)

#现在我们来绘制线性模型以及数据，直观地看出它的拟合。
#np.linspace()在指定的间隔内返回均匀间隔的数字。
x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['Population'], data.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

#由于梯度方程式函数也在每个训练迭代中输出一个代价的向量，所以我们也可以绘制。 请注意，线性回归中的代价函数总是降低的 - 这是凸优化问题的一个例子。
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(epoch), cost, 'r')  # np.arange()返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()



















