import numpy as np


class ActivationStep:

    def forward(self, inputs):

        self.output = np.heaviside(inputs, 0)
