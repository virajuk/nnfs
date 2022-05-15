import numpy as np

from logs import get_logger
logger = get_logger()


# RELU Activation
class ActivationReLU:

    # Forward pass
    def forward(self, inputs):

        # calculate output values from input
        self.output = np.maximum(0, inputs)
