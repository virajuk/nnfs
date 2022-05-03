import math

import numpy as np

softmax_output = [0.7, 0.1, 0.2]


target_output = [1, 0, 0]

# loss = -(math.log(softmax_output[0])*target_output[0] +
#          math.log(softmax_output[1])*target_output[1] +
#          math.log(softmax_output[2])*target_output[2])

# print(loss)
# print(math.log10(100))

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08],
                            [0.2, 0.09, 0.71]])

class_targets = np.array([0, 1, 1, 2])

rows, cols = softmax_outputs.shape
# print(np.sum(class_targets, axis=0))

# for target_idx, distribution in zip(class_targets, softmax_outputs):
#     print(distribution[target_idx])

# print(softmax_outputs[[x for x in range(0, rows)], class_targets])
print(softmax_outputs[range(len(softmax_outputs)), class_targets])

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(neg_log)

average_loss = np.mean(neg_log)
print(average_loss)
