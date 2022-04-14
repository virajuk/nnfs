import numpy as np

import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

print(X[99])
print(y[99])

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.show()
