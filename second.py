import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 2.051, 0.026]]

# layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
print(exp_values)

denominator = np.sum(exp_values, axis=1, keepdims=True)
print(denominator)

print("=============================================================")

exp_values = np.exp(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True))
print(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True))
print(exp_values)

denominator = np.sum(exp_values, axis=1, keepdims=True)
print(denominator)

# softmax00 = numerator / denominator
# print(softmax00)
