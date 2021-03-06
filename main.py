import numpy as np
from mylib import LayerDense

inputs = np.array([1, 2, 3, 2.5])
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])
biases = np.array([2])

# print(inputs.shape)
# print(weights.shape)

outputs = np.dot(inputs, weights.T) + biases
# print(type(outputs))
print(outputs)

dense1 = LayerDense(2, 3)
