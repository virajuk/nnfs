import numpy as np


class ActivationStep:

    def forward(self, inputs):

        return np.heaviside(inputs, 0)
