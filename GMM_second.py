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


class GMM_second(object):
    def __init__(self, n_clusters, max_iter=100):
        super(GMM_second, self).__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.Mu = None
        self.Var = None
        self.Pi = None
        self.W = None
        self.fitted = False

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
            for j in range(self.n_clusters):
                self.W[:, j] = self.Pi[j] * multivariate_normal.pdf(x=data, mean=self.Mu[j], cov=self.Var[j])
            self.W = self.W / self.W.sum(axis=1).reshape(-1, 1)
            Nk = self.W.sum(axis=0)
            self.Mu = np.asarray(
                [np.dot(self.W[:, j].reshape(1, -1), data) / Nk[j] for j in range(self.n_clusters)]).squeeze()
            self.Var = np.asarray([np.dot((data - self.Mu[j]).T,
                                          np.dot(np.diag(self.W[:, j]),
                                                 data - self.Mu[j]
                                                 )) / Nk[j]
                                  for j in range(self.n_clusters) ]
                                  )
            self.Pi = np.asarray(Nk / data.shape[0])
        self.fitted = True
        # 屏蔽结束

    def predict(self, data):
        # 屏蔽开始
        result = []
        if not self.fitted:
            print("not fitted, please fit it first")
            return result
        post = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            post[:, i] = self.Pi[i] * multivariate_normal.pdf(x=data, mean=self.Mu[i], cov=self.Var[i])
        post = post / post.sum(axis=1).reshape(-1, 1)
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
    # plt.figure(figsize=(10, 8))
    # plt.axis([-10, 15, -5, 15])
    # plt.scatter(X1[:, 0], X1[:, 1], s=5)
    # plt.scatter(X2[:, 0], X2[:, 1], s=5)
    # plt.scatter(X3[:, 0], X3[:, 1], s=5)
    # plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM_second(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化
