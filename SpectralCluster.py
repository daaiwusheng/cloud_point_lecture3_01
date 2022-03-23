import numpy as np
from sklearn import cluster
from sklearn.neighbors import KDTree
import scipy
import matplotlib.pyplot as plt

def calculate_distance(point_0, point_1):
    distance = np.sqrt(np.power((point_0 - point_1), 2).sum())
    return distance


def calculate_dist_matrix(data):
    n = len(data)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = dist_matrix[j, i] = calculate_distance(data[i], data[j])
    return dist_matrix


def get_neighbor_graph(data, k):
    n = len(data)
    dist_matrix = calculate_dist_matrix(data)
    W = np.zeros((n, n))
    for i, row_item in enumerate(dist_matrix):
        index_array = np.argsort(row_item)
        # 前k个, 不能报考值为0的,而第一个是自己和自己的距离,一定是0,所以要让开
        W[i][index_array[1: k + 1]] = 1

    W = (W.T + W) / 2
    return W


def get_neighbor_graph_by_kdTree(data, k, sigma=1.0):
    n = len(data)
    W = np.zeros((n, n))
    leaf_size = 2
    kd_tree = KDTree(data, leaf_size)
    for i in range(n):
        query = data[i, :]
        _, indexes = kd_tree.query([query], k)
        for j in indexes:
            W[i, j] = np.exp(np.linalg.norm(data[i] - data[j]) / (-2 * sigma * sigma))
    W = (W.T + W) / 2
    return W


class SpectralCluster(object):
    def __init__(self, n_clusters):
        super(SpectralCluster, self).__init__()
        self.n_clusters = n_clusters
        self.W = None
        self.D = None
        self.L = None
        self.Dn = None
        self.Lsym = None
        self.V = None
        self.need_normalize = True
        self.results = None

    def fit(self, data):
        self.W = get_neighbor_graph_by_kdTree(data, k=6)
        self.D = np.diag(np.sum(self.W, axis=1))
        self.L = self.D - self.W
        self.Dn = np.power(np.linalg.matrix_power(self.D, -1), 0.5)
        self.Lsym = np.dot(np.dot(self.Dn,self.L), self.Dn)

        if self.need_normalize:
            _, self.V = scipy.linalg.eigh(self.Lsym, eigvals=(0, self.n_clusters-1))
        else:
            _, self.V = scipy.linalg.eigh(self.L, eigvals=(0, self.n_clusters - 1))
        kmeans = cluster.KMeans(n_clusters=self.n_clusters)
        kmeans.fit(self.V)
        self.results = kmeans.predict(self.V)

    def predict(self, data):
        return self.results

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

    spect = SpectralCluster(n_clusters=3)
    spect.fit(X)
    cat = spect.predict(X)
    print(cat)
    # 初始化


