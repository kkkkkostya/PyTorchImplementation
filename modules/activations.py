import numpy as np
from .base import Module
from scipy.special import softmax
from scipy.special import log_softmax


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, np.zeros(input.shape))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * np.maximum(np.sign(input), np.zeros(input.shape))


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return 1 / (1 + np.exp(-1 * input))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * 1 / (1 + np.exp(-1 * input)) * (1 - 1 / (1 + np.exp(-1 * input)))


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        out = np.zeros(0)
        for i in range(len(input)):
            softmax_vals = softmax(input[i])
            s = softmax_vals.reshape(-1, 1)
            t = ((np.diagflat(s) - np.dot(s, s.T)) * grad_output[i].reshape(-1, 1)).sum(axis=0)
            if i == 0:
                out = t
                out = np.expand_dims(out, axis=0)
            else:
                t = np.expand_dims(t, axis=0)
                out = np.concatenate((out, t), axis=0)
        return out


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        out = np.zeros(0)
        for i in range(len(input)):
            softmax_vals = softmax(input[i])
            s = np.expand_dims(softmax_vals,axis=0)
            t = ((np.eye(input.shape[1]) - s) * grad_output[i].reshape(-1, 1)).sum(axis=0)
            if i == 0:
                out = t
                out = np.expand_dims(out, axis=0)
            else:
                t = np.expand_dims(t, axis=0)
                out = np.concatenate((out, t), axis=0)
        return out
