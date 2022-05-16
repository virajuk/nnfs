import numpy as np

from logs import get_logger
logger = get_logger('my_app')


class Loss:

    def calculate(self, output, y):

        # logger.info("TESTING LOG")
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)
        logger.info(f"{self.__class__.__name__} Layer output : {data_loss}")

        return data_loss


class LossCategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:

            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:

            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        neg_log_likelihoods = -np.log(correct_confidences)

        return neg_log_likelihoods
