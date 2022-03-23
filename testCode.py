

import numpy as np
from sklearn.neighbors import KDTree
np.random.seed(0)
X = np.random.random((5, 2))  # 5 points in 2 dimensions
tree = KDTree(X, leaf_size=3)
nearest_dist, nearest_ind = tree.query([X[0, :]], k=2)  # k=2 nearest neighbors where k1 = identity
print(X)
print(nearest_dist[:, 1])    # drop id; assumes sorted -> see args!
print(nearest_ind[:, 1])     # drop id

