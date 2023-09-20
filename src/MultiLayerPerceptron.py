import numpy as np
from enum import Enum

class ActivationFunction(Enum):
    RELU = lambda x: x if x > 0 else 0 
    TANH = lambda x: np.tanh(x) 
    LOGISTIC = lambda x: 1 / (1 + np.exp(-x))


class MLP:
    self.input_size=4
    self.input_layer=(ActivationFunction.LOGISTIC, 5)
    self.hidden_layers=[()]
    self.output_layer=(ActivationFunction.RELU, 5)
    def __init__(self, input_size, input_layer, hidden_layers, output_layer):
        self.input_size = input_size
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

if __name__ == '__main__':

