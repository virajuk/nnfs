import numpy as np


# Sigmoid activation
class ActivationSigmoid:

    # forward pass
    def forward(self, inputs):

        # calculate output values from input
        self.output = 1/(1 + np.exp(-inputs))
