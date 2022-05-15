import numpy as np

from logs import get_logger
logger = get_logger('my_app')


# Softmax activation
class ActivationSoftmax:

    # forward pass
    def forward(self, inputs):
        # un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
