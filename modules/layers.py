import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        self.out = input @ self.weight.T
        self.out += self.bias if self.bias is not None else 0
        return self.out

        # return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        if grad_output.size != 0:
            self.grad_inp = grad_output @ self.weight
            return self.grad_inp
        return np.zeros_like(input)
        # return super().compute_grad_input(input, grad_output)

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += (grad_output.T @ input)
        if self.grad_bias is not None and grad_output.size != 0:
            self.grad_bias += np.sum(grad_output.T, axis=1)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            self.mean = np.mean(input, axis=0) if input.size != 0 else 0
            self.var = np.var(input, axis=0) if input.size != 0 else 0
            input = (input - self.mean) / np.sqrt(self.var + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * self.var * input.shape[0] / (input.shape[0] - 1)
            self.norm_input = input
            if self.affine:
                input = input * self.weight + self.bias
        else:
            input = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.norm_input = input
            if self.affine:
                input = input * self.weight + self.bias

        return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if grad_output.size == 0:
            return np.zeros_like(input)
        N, D = grad_output.shape
        if self.training:
            var_plus_eps = self.var + self.eps
            gamma = self.weight
            x_ = self.norm_input

            # calculate gradients
            if self.affine:
                dx_ = np.matmul(np.ones((N, 1)), gamma.reshape((1, -1))) * grad_output
            else:
                dx_ = grad_output
            dx = N * dx_ - np.sum(dx_, axis=0) - x_ * np.sum(dx_ * x_, axis=0)
            dx *= (1.0 / N) / np.sqrt(var_plus_eps)
        else:
            var_plus_eps = self.running_var + self.eps
            dx = np.full(input.shape, fill_value=1 / np.sqrt(var_plus_eps))
            if self.affine:
                dx *= self.weight
            dx *= grad_output

        return dx

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.affine:
            x_ = self.norm_input
            dgamma = np.sum(x_ * grad_output, axis=0)
            dbeta = np.sum(grad_output, axis=0)

            self.grad_weight += dgamma
            self.grad_bias += dbeta

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """

    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, input.shape)
            return 1 / (1 - self.p) * self.mask * input
        else:
            return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            return grad_output * 1 / (1 - self.p) * self.mask
        else:
            return grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """

    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        self.inputs = {}
        if len(self.modules) == 0:
            return np.zeros(input.shape)
        output = self.modules[0](input)
        self.inputs[0] = output
        for i in range(1, len(self.modules)):
            output = self.modules[i](output)
            self.inputs[i] = output
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """

        # forward
        inputs = {}
        if len(self.modules) == 0:
            return np.zeros(input.shape)
        inputs[0] = input
        output = self.modules[0](input)
        inputs[1] = output
        for i in range(1, len(self.modules) - 1):
            output = self.modules[i](output)
            inputs[i + 1] = output

        # backward
        if len(self.modules) == 0:
            return np.zeros(input.shape)
        output = self.modules[-1].backward(inputs[len(inputs) - 1], grad_output)
        for i in range(len(self.modules) - 2, -1, -1):
            output = self.modules[i].backward(inputs[i], output)
        return output

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
