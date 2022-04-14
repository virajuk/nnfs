import numpy as np

a = [1, 2, 3]
print(a)

arr = np.array([a])
print(arr)

new_a = np.expand_dims(np.array(a), axis=0)
print(new_a)
print(new_a.shape)
