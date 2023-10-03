from enum import Enum
from typing import Callable

import numpy as np
import math


activationFunc = Callable[[np.ndarray], np.ndarray]

class ActivationFunction(Enum):
    RELU = lambda x: np.maximum(0, x)
    TANH = lambda x: np.tanh(x)
    SIGMOID = lambda x: 1 / (1 + np.exp(-x))
    CEIL = lambda x: math.ceil(x)