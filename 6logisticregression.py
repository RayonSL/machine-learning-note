# 1 Logistic regression 在这部分的练习中，你将建立一个逻辑回归模型来预测一个学生是否能进入大学。
# 假设你是一所大学的行政管理人员，你想根据两门考试的结果，来决定每个申请人是否被录取。
# 你有以前申请人的历史数据，可以将其用作逻辑回归训练集。
# 对于每一个训练样本，你有申请人两次测评的分数以及录取的结果。
# 为了完成这个预测任务，我们准备构建一个可以基于两次测试评分来评估录取可能性的分类模型。
# 1.1 Visualizing the data 在开始实现任何学习算法之前，如果可能的话，最好将数据可视化。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:\ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
data.head()
data.describe()

#让我们创建两个分数的散点图，并使用颜色编码来可视化，如果样本是正的（被接纳）或负的（未被接纳）。

positive = data[data.admitted.isin(['1'])]  # 1
negetive = data[data.admitted.isin(['0'])]  # 0

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')
ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
# 设置图例显示在图的上方
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12),ncol=3)
# 设置横纵坐标名
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(- z))
#让我们做一个快速的检查，来确保它可以工作。
x1 = np.arange(-10, 10, 0.1)
plt.plot(x1, sigmoid(x1), c='m')#c表示颜色
plt.show()

def cost(theta, X, y):
    first = (-y) * np.log(sigmoid(X @ theta))#@表示X*theta的转置
    second = (1 - y)*np.log(1 - sigmoid(X @ theta))
    return np.mean(first - second)

#现在，我们要做一些设置，获取我们的训练集数据。
# add a ones column - this makes the matrix multiplication work out easier
if 'Ones' not in data.columns:#如果数据列中没有全1的列
    data.insert(0, 'Ones', 1)#全1列插入第0列，列名为ones

# set X (training data) and y (target variable)
X = data.iloc[:, :-1].as_matrix()  # Convert the frame to its Numpy-array representation.
y = data.iloc[:, -1].as_matrix()  # Return is NOT a Numpy-matrix, rather, a Numpy-array.

theta = np.zeros(X.shape[1])#1表示取出X的列长度，若是0表示取出X的行长度

#让我们来检查矩阵的维度，确保一切良好。
X.shape, theta.shape, y.shape
# ((100, 3), (3,), (100,))
cost(theta, X, y)
# 0.6931471805599453


#批量梯度下降（batch gradient descent）
def gradient(theta, X, y):
    return (X.T @ (sigmoid(X @ theta) - y))/len(X)
# the gradient of the cost is a vector of the same length as θ where the jth element (for j = 0, 1, . . . , n)
gradient(theta, X, y)
# array([ -0.1, -12.00921659, -11.26284221])
# 1.5 Learning θ parameters 注意，我们实际上没有在这个函数中执行梯度下降，我们仅仅在计算梯度。
# 在练习中，一个称为“fminunc”的Octave函数是用来优化函数来计算成本和梯度参数。
# 由于我们使用Python，我们可以用SciPy的“optimize”命名空间来做同样的事情。
# 这里我们使用的是高级优化算法，运行速度通常远远超过梯度下降。方便快捷。
# 只需传入cost函数，已经所求的变量theta，和梯度。
# cost函数定义变量时变量tehta要放在第一个，若cost函数只返回cost，则设置fprime=gradient。
import scipy.optimize as opt
#这里使用fimin_tnc或者minimize方法来拟合，minimize中method可以选择不同的算法来计算，其中包括TNC
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
result
# (array([-25.16131867,   0.20623159,   0.20147149]), 36, 0)

#下面是第二种方法，结果是一样的
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='TNC', jac=gradient)
res
# help(opt.minimize)
# res.x  # final_theta
cost(result[0], X, y)

#学习好了参数θ后，我们来用这个模型预测某个学生是否能被录取。
#接下来，我们需要编写一个函数，用我们所学的参数theta来为数据集X输出预测。
# 然后，我们可以使用这个函数来给我们的分类器的训练精度打分。
#当hθ大于等于0.5时，预测 y=1
#当hθ小于0.5时，预测 y=0 。

def predict(theta, X):
    probability = sigmoid(X@theta)
    return [1 if x >= 0.5 else 0 for x in probability]  # return a list

final_theta = result[0]
predictions = predict(final_theta, X)
correct = [1 if a==b else 0 for (a, b) in zip(predictions, y)]
#zip()函数分别从a和b依次各取出一个元素组成元组，再将依次组成的元组组合成一个新的迭代器--新的zip类型数据
# example:
# m = [[1,2,3], [4,5,6], [7,8,9]]
# n = [[2,2,2], [3,3,3], [4,4,4]]
# zip(m, n)将返回([1, 2, 3], [2, 2, 2]), ([4, 5, 6], [3, 3, 3]), ([7, 8, 9], [4, 4, 4])

accuracy = sum(correct) / len(X)
print(accuracy)

#> 0.89 可以看到我们预测精度达到了89%，not bad. 也可以用skearn中的方法来检验。
from sklearn.metrics import classification_report
print(classification_report(predictions, y))

#6.3决策边界
#X×θ=0 (this is the line)
#θ0+x1θ1+x2θ2=0
x1 = np.arange(130, step=0.1)
x2 = -(final_theta[0] + x1*final_theta[1]) / final_theta[2]

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')#scatter绘制散点图中c表示颜色，marker表示markershtyle
ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.plot(x1, x2)#画直线
ax.set_xlim(0, 130)#设置x轴范围
ax.set_ylim(0, 130)#设置y轴范围
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Decision Boundary')
plt.show()

"""
Regularized logistic regression
在训练的第二部分，我们将要通过加入正则项提升逻辑回归算法。
简而言之，正则化是成本函数中的一个术语，它使算法更倾向于“更简单”的模型
（在这种情况下，模型将更小的系数）。这个理论助于减少过拟合，提高模型的泛化能力。

设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果。
对于这两次测试，你想决定是否芯片要被接受或抛弃。
为了帮助你做出艰难的决定，你拥有过去芯片的测试数据集，从其中你可以构建一个逻辑回归模型。
"""
#读入数据
data2 = pd.read_csv('D:\ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
data2.head()

def plot_data():
    positive = data2[data2['Accepted'].isin([1])]#.isin表示检测1是不是在这个里面
    negative = data2[data2['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(8,5))#fig绘制面板，AXES表示一个图表，AXIS表示坐标轴
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    plt.show()
plot_data()

#注意到其中的正负两类数据并没有线性的决策界限。
#因此直接用logistic回归在这个数据集上并不能表现良好，因为它只能用来寻找一个线性的决策边界。
#所以接下会提到一个新的方法。
#2.2 Feature mapping 一个拟合数据的更好的方法是从每个数据点创建更多的特征。
#我们将把这些特征映射到所有的x1和x2的多项式项上，直到第六次幂。

def feature_mapping(x1, x2, power):
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)
            #
#     data = {"f{}{}".format(i - p, p): np.power(x1, i - p) * np.power(x2, p)
#                 for i in np.arange(power + 1)
#                 for p in np.arange(i + 1)
#             }
    return pd.DataFrame(data)

x1 = data2['Test 1'].as_matrix()
x2 = data2['Test 2'].as_matrix()

_data2 = feature_mapping(x1, x2, power=6)
_data2.head()#返回DataFrame.head(n=5)Return the first n rows.[n默认为5]
print(_data2.head())

#正则化逻辑回归（regularized logistic gradient）
#先获取特征，标签以及参数theta，确保维度良好。
# 这里因为做特征映射的时候已经添加了偏置项，所以不用手动添加了。
X = _data2.as_matrix()
y = data2['Accepted'].as_matrix()
theta = np.zeros(X.shape[1])#1表示列数，0表示行数
  # ((118, 28), (118,), (28,))
print(X.shape, y.shape, theta.shape)
def costReg(theta, X, y, l=1):
    # 不惩罚第一项
    _theta = theta[1: ]
    reg = (l / (2 * len(X))) * (_theta @ _theta)  # _theta@_theta == inner product
    return cost(theta, X, y) + reg

print(costReg(theta,X,y,l=1))

def gradientReg(theta,X,y,l=1):
    reg = (1/len(X))*theta
    reg[0] = 0
    return gradient(theta,X,y)+reg

print(gradientReg(theta,X,y,1))

result2 = opt.fmin_tnc(func=costReg,x0=theta,fprime=gradientReg,args=(X,y,2))#func请求补偿的函数，x0表示该函数中的哪个
# 元素，fprime为func函数的梯度，args为元组

#评估逻辑回归函数
final_theta = result2[0]
predictions = predict(final_theta, X)
correct = [1 if a==b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
print(accuracy)

#决策边界
x = np.linspace(-1, 1.5, 250)#.linespace(start,stop,num)在指定的间隔内返回均匀的数字
xx, yy = np.meshgrid(x, x)#生成以某点为中心指定半径内的数值矩阵

z = feature_mapping(xx.ravel(), yy.ravel(), 6).as_matrix()
z = z @ final_theta
z = z.reshape(xx.shape)

plot_data()
plt.contour(xx, yy, z, 0)
plt.ylim(-.8, 1.2)



