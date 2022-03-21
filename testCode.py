import numpy as np

data = np.arange(1, 7).reshape(-1, 2)
print(data)

print(data ** 2)
print(np.dot(data.T, data))

weights = np.array([1, 2, 1])
t = np.average(a=data ** 2, axis=0, weights=weights)
print(t)

Nk = weights.sum(axis=0)
Var = np.asarray([np.dot(data.T,
                         np.dot(np.diag([weights[j]]),
                                data
                                )) / Nk[j]
                  for j in range(3)]
                 )
print("=" * 30)
print(Var)
