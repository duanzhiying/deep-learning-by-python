# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist


# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# mini-batch小批量抽取
train_size = x_train.shape[0]
batch_size = 100
batch_mask = np.random.choice(train_size, batch_size)
x_train_batch = x_train[batch_mask]
t_train_batch = t_train[batch_mask]

# 偏导数计算
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
# 基于x0和x1生成网格点坐标矩阵
X, Y = np.meshgrid(x0, x1)


X = X.flatten()
Y = Y.flatten()

x_total = np.array([X,Y])

f_vaue = function_2(x_total)

