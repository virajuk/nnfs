import numpy as np


# Step function activation
class ActivationStep:

    # forward pass
    def forward(self, inputs):

        # calculate output values from input
        self.output = np.heaviside(inputs, 0)
