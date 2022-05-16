import numpy as np
from mylib import LayerDense

a = [1, 2, 3]
# print(a)

arr = np.array([a])
# print(arr)

new_a = np.expand_dims(np.array(a), axis=0)
# print(new_a)
# print(new_a.shape)

a = np.array([1 + 2j, 3 + 4j])
b = np.array([5 + 6j, 7 + 8j])

ab = np.vdot(a, b)
print(ab)

ba = np.vdot(b, a)
print(ba)
