import numpy as np


class ActivationSigmoid:

    def forward(self, inputs):

        self.output = 1/(1 + np.exp(-inputs))
