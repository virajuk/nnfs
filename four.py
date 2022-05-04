import math

import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08],
                            [0.2, 0.09, 0.71]])

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

# class_targets = np.array([0, 1, 1, 2])

if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs*class_targets, axis=1)

print(correct_confidences)

neg_log = -np.log(correct_confidences)
print(neg_log)

average_loss = np.mean(neg_log)
print(average_loss)
