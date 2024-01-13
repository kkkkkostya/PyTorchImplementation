import numpy as np
from typing import Tuple
from .base import Module, Optimizer


class SGD(Optimizer):
    """
    Optimizer implementing stochastic gradient descent with momentum
    """

    def __init__(self, module: Module, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param momentum: momentum coefficient (alpha)
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]

        k = 0
        for param, grad, m in zip(parameters, gradients, self.state['m']):
            if self.weight_decay != 0:
                grad += self.weight_decay * param
            m = self.momentum * m + grad
            gradients[k] = grad
            self.state['m'][k] = m
            parameters[k] -= self.lr * m
            k += 1
        """
        your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
          - update momentum variable (m)
          - update parameter variable (param)
        hint: consider using np.add(..., out=m) for in place addition,
          i.e. we need to change original array, not its copy
        """


class Adam(Optimizer):
    """
    Optimizer implementing Adam
    """

    def __init__(self, module: Module, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param betas: Adam beta1 and beta2
        :param eps: Adam eps
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]
            self.state['v'] = [np.zeros_like(param) for param in parameters]
            self.state['t'] = 0

        self.state['t'] += 1
        t = self.state['t']
        k = 0
        for param, grad, m, v in zip(parameters, gradients, self.state['m'], self.state['v']):
            if self.weight_decay != 0:
                grad += self.weight_decay * param
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_norm = m / (1 - self.beta1 ** t)
            v_norm = v / (1 - self.beta2 ** t)
            gradients[k] = grad
            self.state['m'][k] = m
            self.state['v'][k] = v
            parameters[k] -= self.lr * (m_norm / (np.sqrt(v_norm) + self.eps))
            k += 1

            """
            your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
              - update first moment variable (m)
              - update second moment variable (v)
              - update parameter variable (param)
            hint: consider using np.add(..., out=m) for in place addition,
              i.e. we need to change original array, not its copy
            """
