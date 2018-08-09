#待训练数据A、B为自变量，C为因变量。
#在写程序之前，要先导入我们需要的模块。
import numpy as np
from numpy import genfromtxt #genfromtxt主要执行两个循环运算。第一个循环将文件的每一行转换成字符串序列。第二个循环将每个字符串序列转换为相应的数据类型。

#首先将数据读入Python中，程序如下所示：
dataPath = r"D:\house.csv"
dataSet = genfromtxt(dataPath,delimiter = ',')#delimiter:str,int,or sequence,optional.他是分割值，即表示你的数组用什么来分割。

#接下来将读取的数据分别得到自变量矩阵和因变量矩阵：
#这里需要注意的是，在原有自变量的基础上，需要主观添加一个均为1的偏移量，
#即公式中的x0。原始数据的前n-1列再加上添加的偏移量组成自变量trainData，
#最后一列为因变量trainLabel。
def getData(dataSet):
    m,n = np.shape(dataSet)#快速读取矩阵的形状，这里是10*3，所以m=10，n=3
    trainData = np.ones((m,n))#产生一个10*3的数组，数组每个元素都是1
    trainData[:,:-1] = dataSet[:,:-1]#第一个：表示第一：表示行，取所有行，-1表示到最后一列（不包括最后一列）
    trainLabel = dataSet[:,-1]#-1表示最后一列
    return trainData,trainLabel

#下面开始实现批处理梯度下降算法：
#x为自变量训练集，y为自变量对应的因变量训练集；
#theta为待求解的权重值，需要事先进行初始化；a
#lpha是学习率；m为样本总数；maxIterations为最大迭代次数；

def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()#自变量矩阵转置
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)#np.dot(A, B)：对于二维矩阵，计算真正意义上的矩阵乘积，同线性代数中矩阵乘法的定义。对于一维矩阵，计算两者的内积。
        loss = hypothesis - y
        # print loss
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
    return theta

#求解权重过程，初始化batchGradientDescent函数需要的各个参数：
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)
theta = np.ones(n)
alpha = 0.05
maxIteration = 1000

#alpha和maxIterations可以更改，之后带入到batchGradientDescent中可以求出最终权重值。
theta = batchGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)

#之后我们给出一组数据，需要进行预测，预测函数：
#x为待预测值的自变量，thta为已经求解出的权重值，yPre为预测结果
def predict(x,theta):
    m,n = np.shape(x)
    xTest = np.ones((m,n+1))
    xTest[:,:-1] = x
    yPre = np.dot(xTest,theta)
    return yPre

#对该组数据进行预测，程序如下：
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print (predict(x, theta))





