import numpy as np


# RELU Activation
class ActivationReLU:

    # Forward pass
    def forward(self, inputs):

        self.output = np.maximum(0, inputs)
