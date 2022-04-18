import numpy as np


# RELU Activation
class ActivationReLU:

    def forward(self, inputs):

        self.output = np.maximum(0, inputs)
