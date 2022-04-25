import math
import numpy as np

import nnfs

# layer_outputs = [4.8, 1.21, 2.385]

E = math.e

# print(E)

layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])

result = np.sum(layer_outputs, axis=0, keepdims=True)

print(result.shape)
print(result)
