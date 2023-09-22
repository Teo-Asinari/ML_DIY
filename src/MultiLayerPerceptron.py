import numpy as np
from enum import Enum


class ActivationFunction(Enum):
    RELU = lambda x: np.maximum(0, x)
    TANH = lambda x: np.tanh(x)
    LOGISTIC = lambda x: 1 / (1 + np.exp(-x))


class MLP:
    input_size = 4
    input_layer_schema = (ActivationFunction.LOGISTIC, 5)
    hidden_layer_schemas = [()]
    output_layer_schema = (ActivationFunction.RELU, 5)
    layers = []

    def __init__(self, input_size, layer_schemas):
        self.input_size = input_size
        self.layer_schemas = layer_schemas

    def initLayers(self):
        self.layers = []
        self.layers.append(np.random.rand(self.layer_schemas[0][1], self.input_size))

        prev_layer_size = self.layer_schemas[0][1]

        for layer in self.layer_schemas[1:]:
            self.layers.append(np.random.rand(layer[1], prev_layer_size))
            prev_layer_size = layer[1]

        print("These are the layers: " + str(self.layers))

    def forwardPass(self, input_vec):
         currVal = input_vec
         print("curr_val start is: " + str(currVal))
         for layerIdx in range(len(self.layers)):
             currActivationFunc = self.layer_schemas[layerIdx][0]
             currActivation = np.dot(self.layers[layerIdx], currVal)
             print("curr_val before activation func is: " + str(currActivation))
             currVal = currActivationFunc(currActivation)
             print("curr_val after activation func is: " + str(currVal))
         print("Final curr_val is: " + str(currVal))
         return currVal


if __name__ == '__main__':
    print("Begin construct basic MLP")
    layerSchemas = [(ActivationFunction.LOGISTIC, 5), (ActivationFunction.TANH, 5),
                          (ActivationFunction.TANH, 5), (ActivationFunction.RELU, 5)]
    basicMLP = MLP(5, layerSchemas)
    basicMLP.initLayers()
    basicMLP.forwardPass(np.array([1,2,3,4,5]))
