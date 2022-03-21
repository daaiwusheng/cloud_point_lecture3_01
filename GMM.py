# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random, math
import KMeans
from sklearn import cluster
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=100):
        super(GMM, self).__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.Mu = None
        self.Var = None
        self.Pi = None
        self.W = None
        self.fitted = False

    # 屏蔽开始
    # 更新W
    def update_W(self, data, Mu, Var, Pi):
        n_points = len(data)
        pdfs = np.zeros((n_points, self.n_clusters))
        for i in range(self.n_clusters):
            pdfs[:, i] = Pi[i] * multivariate_normal.pdf(data, Mu[i], Var[i])
        W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return W

    # 计算W
    def calculate_W(self, data):
        n_points = len(data)
        pdfs = np.zeros((n_points, self.n_clusters))
        for i in range(self.n_clusters):
            t = self.Var[i]
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(x=data, mean=self.Mu[i], cov=self.Var[i])
        W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return W

    # 更新pi
    def update_Pi(self, W):
        Pi = W.sum(axis=0) / W.sum()
        return Pi

    # 更新Mu
    def update_Mu(self, W, Mu, data):
        for i in range(Mu.shape[0]):
            Mu[i] = np.average(data, axis=0, weights=W[:, i])
        return Mu

    # 更新Var
    def update_Var(self, data, W, Mu, Var):
        Nk = self.W.sum(axis=0)
        for i in range(self.n_clusters):
            # 注意这里是需要的是协方差矩阵
            Var[i] = np.dot((data - Mu[i]).T, np.dot(np.diag(W[:, i]), data - Mu[i])) / Nk[i]
        return Var

    # 屏蔽结束

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        # k_means = KMeans.K_Means(n_clusters=self.n_clusters)
        k_means = cluster.KMeans(n_clusters=self.n_clusters)
        k_means.fit(data)
        self.Mu = np.asarray(k_means.cluster_centers_)
        self.Var = np.asarray([np.eye(data.shape[1])] * self.n_clusters)
        self.Pi = np.asarray([1 / self.n_clusters] * self.n_clusters).reshape(self.n_clusters, 1)
        self.W = np.zeros((data.shape[0], self.n_clusters))

        for i in range(self.max_iter):
            self.W = self.update_W(data, self.Mu, self.Var, self.Pi)
            self.Pi = self.update_Pi(self.W)
            self.Mu = self.update_Mu(self.W, self.Mu, data)
            self.Var = self.update_Var(data, self.W, self.Mu, self.Var)
        self.fitted = True
        # 屏蔽结束

    def predict(self, data):
        # 屏蔽开始
        result = []
        if not self.fitted:
            print("not fitted, please fit it first")
            return result
        post = self.calculate_W(data)
        result = np.argmax(post, axis=1)
        return result
        # 屏蔽结束


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化
