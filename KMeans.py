# 文件功能： 实现 K-Means 算法

import numpy as np


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers_ = None

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        m = data.shape[0]
        # 在数据集中随机选择 k 个点,作为初始中心点
        self.centers_ = data[np.random.choice(m, self.k_, replace=False), :]
        last_loss = 0
        for i in range(self.max_iter_):
            distance_matrix = np.empty((m, self.k_))  # 到达中心点的距离矩阵
            for j, center in enumerate(self.centers_):
                # axis 为1 表示along the column, 就是行向量, 这样得到的就是每个点到当前中心点的欧氏距离
                distance_matrix[:, j] = np.linalg.norm(data - center, axis=1)
            # 下面, 多少个中心点,那么每个点就会计算多少次距离, 但是要沿着列轴, 比较最短距离,作为label
            labels = np.argmin(distance_matrix, axis=1)
            # 每个点最短距离只和作为loss
            current_loss = (np.min(distance_matrix, axis=1) ** 2).sum()
            if np.abs(current_loss - last_loss) < self.tolerance_:
                break
            last_loss = current_loss
            # 下面重新计算中心点坐标
            for ck in range(self.k_):
                # 这里ck索引,恰好就是数据的label, 所以把对应label的点全部选出来,
                # 一个class中的所有点在计算一次中心点坐标
                ck_points = data[labels == ck]
                self.centers_[ck] = np.mean(ck_points, axis=0)
        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        distance_matrix = np.empty((p_datas.shape[0], self.k_))
        for center_index, center_point in enumerate(self.centers_):
            distance_matrix[:, center_index] = np.linalg.norm(p_datas - center_point, axis=1)
        result = np.argmin(distance_matrix, axis=1).tolist()
        # 屏蔽结束
        return result


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)
