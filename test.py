import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y

X,y = load_data('D:\ex3data1.mat')
print(np.unique(y))#.unique查看y中有几种不同的值
print(X.shape, y.shape)

def plot_an_image(X):
    pick_one = np.random.randint(0, 5000)
    #random.randint(a, b)     # 返回闭区间 [a, b] 范围内的整数值
    #numpy.random.randint(a, b)   # 返回开区间 [a, b) 范围内的随机整数值
    image = X[pick_one,:]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)),cmap='gray_r')
    plt.xticks([])#去除刻度，美观作用
    plt.yticks([])
    plt.show()
    print('Which number {}'.format(y[pick_one]))

plot_an_image(X)

def plot_100_image(X):
    sample_index = np.random.choice(np.arange(X.shape[0]), 100, replace=False)#从X的行中挑选出100个不一样
    print(sample_index)
    sample_image = X[sample_index, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_image[10*row+column].reshape((20, 20)), cmap='gray_r')
    plt.xticks([])  # 去除刻度，美观作用
    plt.yticks([])
    plt.show()
plot_100_image()
