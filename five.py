import math

import numpy as np

import nnfs
from nnfs.datasets import spiral_data

from mylib import LayerDense
from mylib import ActivationReLU
from mylib import ActivationSoftmax
from mylib import LossCategoricalCrossEntropy

nnfs.init()

X, y = spiral_data(samples=150, classes=3)

dense_1 = LayerDense(2, 3)
activation_1 = ActivationReLU()

dense_2 = LayerDense(3, 3)
activation_2 = ActivationSoftmax()

loss_fn = LossCategoricalCrossEntropy()

########################################

dense_1.forward(X)
activation_1.forward(dense_1.output)

dense_2.forward(activation_1.output)
activation_2.forward(dense_2.output)

loss = loss_fn.calculate(activation_2.output, y)
print(f"Loss {loss}")
