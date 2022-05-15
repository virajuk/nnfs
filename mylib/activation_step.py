import numpy as np

from logs import get_logger
logger = get_logger()


# Step function activation
class ActivationStep:

    # forward pass
    def forward(self, inputs):

        # calculate output values from input
        self.output = np.heaviside(inputs, 0)
