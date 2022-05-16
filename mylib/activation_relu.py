import numpy as np

from logs import get_logger
logger = get_logger('my_app')


# RELU Activation
class ActivationReLU:

    # Forward pass
    def forward(self, inputs):

        # calculate output values from input
        self.output = np.maximum(0, inputs)
        logger.info(f"{self.__class__.__name__} Layer output : {self.output.shape}")
