import numpy as np

import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from mylib import LayerDense
from mylib import ActivationStep

# nnfs.init()
#
# X, y = spiral_data(samples=100, classes=3)
#
# dense1 = LayerDense(2, 3)
#
# dense1.forward(X)
#
# print(len(dense1.output))


step = ActivationStep()

# inputs = np.array([-0.2])
# inputs = np.array([-1, 0, 0.3, 2.5])
inputs = np.array([[0, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 2.051, 0.026]])
print(step.forward(inputs))
