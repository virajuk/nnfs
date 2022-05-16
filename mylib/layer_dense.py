import numpy as np
import nnfs

from logs import get_logger
logger = get_logger('my_app')


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        logger.info(f"{self.__class__.__name__} Layer output : {self.output.shape}")
