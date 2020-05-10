# -*- coding: utf-8 -*-#
'''
# Name:         normal_show
# Description:  显示高斯混合模型的一些信息
# Author:       super
# Date:         2020/5/10
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

# 画出聚类图像
def plot_clusters(X, Mu_true, Var_true, ang = 0):
    colors = ['b', 'g', 'r', 'c']
    n_clusters = len(Mu_true)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    ax = plt.gca()
    for i in range(n_clusters):
        plt.scatter(X[i][:, 0], X[i][:, 1], s=5, c=colors[i])
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
        if i == 3:
            ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], angle=20, **plot_args)
        else:
            ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], angle=ang,**plot_args)
        ax.add_patch(ellipse)
    plt.show()

# 画出聚类图像
def plot_cluster(X, Mu_true, Var_true):
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    ax = plt.gca()
    plt.scatter(X[:, 0], X[:, 1], s=5, c='b')
    plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': 'b', 'alpha': 0.5}
    ellipse = Ellipse(Mu_true, 3 * Var_true[0], 3 * Var_true[1], angle=0, **plot_args)
    ax.add_patch(ellipse)
    plt.show()

if __name__ == "__main__":
    nums = [400, 600, 1000, 500]
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7], [9, 4.5]]
    true_Var = [[1, 3], [2, 2], [6, 2], [1, 3]]
    var = [np.diag(true_Var[0]), np.diag(true_Var[1]),np.diag(true_Var[2]),np.array([[1,-1],[-1,3]])]
    print(np.diag(true_Var[0]))
    print(type(np.diag(true_Var[0])))
    a = np.array([[1,2],[3,4]])
    print(a)
    print(type(a))
    # 第一簇的数据
    num1, mu1, var1 = nums[0], true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, var[0], num1)
    plot_cluster(X1, [0.5, 0.5], [1, 3])
    # 第二簇的数据
    num2, mu2, var2 = nums[1], true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, var[1], num2)
    plot_cluster(X2, [5.5, 2.5], [2,2])
    # 第三簇的数据
    num3, mu3, var3 = nums[2], true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, var[2], num3)
    plot_cluster(X3, [1, 7], [6, 2])
    # 第四簇的数据
    num4, mu4, var4 = nums[3], true_Mu[3], true_Var[3]
    X4 = np.random.multivariate_normal(mu4, var[3], num4)
    plot_cluster(X3, [1, 7], [6, 2])
    X = [X1, X2, X3, X4]

    # plot_clusters(X1, X2, X3, X4, true_Mu, true_Var)
    plot_clusters(X, true_Mu, true_Var)