import numpy as np

import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from mylib import LayerDense
from mylib import ActivationStep
from mylib import ActivationSigmoid
from mylib import ActivationReLU

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)

dense1.forward(X)

step = ActivationReLU()

step.forward(dense1.output)

print(dense1.output[100:110])
print(step.output[100:110])
