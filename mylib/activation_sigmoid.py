import numpy as np


class ActivationSigmoid:

    def forward(self, inputs):

        return 1/(1 + np.exp(-inputs))
