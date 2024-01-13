import numpy as np
from .base import Criterion
from .activations import LogSoftmax
from scipy.special import softmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        if input.size != 0:
            return 1 / (input.shape[0] * input.shape[1]) * (((target - input) ** 2).sum(axis=1)).sum()
        return 0

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        if input.size != 0:
            return 1 / (input.shape[0] * input.shape[1]) * (2 * (input - target))
        return np.array([])


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """

    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        log_soft = self.log_softmax(input)
        loss_func = 0
        for i in range(input.shape[0]):
            loss_func += log_soft[i][target[i]]
        if input.size != 0:
            return -1 / input.shape[0] * loss_func
        return 0

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        if input.size != 0:
            ind = np.arange(input.shape[0])
            grad = -1 * softmax(input, axis=1)
            grad[ind, target] = 1 + grad[ind, target]
            grad *= (-1 / input.shape[0])
            return grad
        return np.array([])
