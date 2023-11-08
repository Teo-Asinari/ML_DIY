from enum import Enum
from typing import Callable

import numpy as np


activationFunc = Callable[[np.ndarray], np.ndarray]

class ActivationFunction(Enum):
    RELU = lambda x: np.maximum(0, x)
    RELU_DERIV = lambda x: np.heaviside(x, 1)
    TANH = lambda x: np.tanh(x)
    SIGMOID = lambda x: 1 / (1 + np.exp(-x))
    SIGMOID_DERIV = lambda x: ActivationFunction.SIGMOID(x) * (1 - ActivationFunction.SIGMOID(x))
    ROUND = lambda x: np.round(x)
