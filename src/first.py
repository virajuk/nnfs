import numpy as np

import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from src.layer_dense import LayerDense

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)

dense1.forward(X)

print(dense1.output[:3])