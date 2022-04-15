import numpy as np
import nnfs


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
