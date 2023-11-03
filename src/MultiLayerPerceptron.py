import argparse
import math

import numpy as np

from Utils.DebugUtils import printd
from Utils.ActivationFunction import activationFunc, ActivationFunction

from typing import List, Tuple

layerSchema = Tuple[activationFunc, int]

DEBUG = False
class MLPLayer:
    def __init__(self, inputsize: int = 2, size: int = 2, func: activationFunc =
    ActivationFunction.RELU):
        self.inputsize = inputsize
        self.size = size
        self.func = func
        self.weights = None
        self.activation = None
        self.output = None

class MatrixOutMLPLayer(MLPLayer):
    def __init__(self, inputsize: int = 2, size: int = 2, func: activationFunc =
    ActivationFunction.RELU):
        super.__init__(self, inputsize, size, func)
        self.weights = np.random.rand(size, inputsize)
        self.activation = np.zeros(size, 1)
        self.output = np.zeros(size, 1)

class ScalarOutMLPLayer(MLPLayer):
    def __init__(self, inputsize : int = 2, size: int = 2, func: activationFunc = ActivationFunction.RELU):
        super.__init__(self, inputsize, size, func)
        self.weights = np.random.rand(size, inputsize)
        self.activation = 0
        self.output = 0

class MLP:
    def __init__(self, layers: List[MLPLayer]):
        self.layers = layers
        self.learning_rate = 0.1

    # # Design: Each Layer is a list of [weights matrix,
    # # last activation, activation_func(last activation)]
    # def initLayers(self) -> List[np.ndarray]:
    #     layers = [[np.random.rand(self.layer_schemas[0][1], self.input_size),
    #                0 if self.layer_schemas[0][1] == 1 else np.zeros(self.layer_schemas[0][1], 1),
    #                0 if self.layer_schemas[0][1] == 1 else np.zeros(self.layer_schemas[0][1], 1)]]
    #
    #     prev_layer_size = self.layer_schemas[0][1]
    #
    #     for layerSchema in self.layer_schemas[1:]:
    #         layers.append([np.random.rand(layerSchema[1], prev_layer_size),
    #                        0 if layerSchema[1] == 1 else np.zeros(layerSchema[1], 1),
    #                        0 if layerSchema[1] == 1 else np.zeros(layerSchema[1], 1)])
    #         prev_layer_size = layerSchema[1]
    #
    #     printd("These are the layers: " + str(layers), DEBUG)
    #     return layers

    def forwardPass(self, input_vec: np.ndarray):
        currVal = input_vec
        printd("curr_val start is: " + str(currVal), DEBUG)
        for layer in self.layers:
            currActivationFunc = layer.func
            currActivation = np.dot(layer.weights, currVal)
            layer.activation = currActivation
            printd("curr_val before activation func is: " + str(currActivation), DEBUG)
            currVal = currActivationFunc(currActivation)
            layer.output = currVal
            printd("curr_val after activation func is: " + str(currVal), DEBUG)
        printd("Final curr_val is: " + str(currVal), DEBUG)
        return currVal

# Hard-code the backprop matrices (derived by hand) for a small MLP of known layout
class BasicMLP(MLP):
    def __init__(self):
        self.input_size = 2
        self.layer_schemas = [(ActivationFunction.RELU, 2), (ActivationFunction.RELU, 2),
                    (ActivationFunction.SIGMOID, 1)]
        self.layers = self.initLayers()

    def backPropSingleError(self, datum):
       input = datum[0]
       expected = datum[1]
       actual = self.forwardPass(input)
       error = 0.5 * math.pow((expected - actual), 2)

       dError = 1
       dOutput3 = (expected - actual)
       dLayer3 = dOutput3 * ActivationFunction.SIGMOID_DERIV(self.layers[2].activation)
       dW3 = dLayer3 * self.layers[1].output
       dOutput2 = dLayer3 * self.layers[2].weights
       dLayer2 = np.multiply(dOutput2, ActivationFunction.RELU_DERIV(self.layers[1].activation)) # hadamard/elementwise product
       dW2 = np.matmul(dLayer2, np.transpose(self.layers[0].output))
       dOutput1 = np.matmul(np.transpose(self.layers[1].weights), dLayer2)
       dLayer1 = np.multiply(dOutput1, ActivationFunction.RELU_DERIV(self.layers[0].activation))
       dW1 = np.matmul(dLayer1, np.transpose(input))

       self.layers[2][0] -= self.learning_rate * dW3
       self.layers[1][0] -= self.learning_rate * dW2
       self.layers[0][0] -= self.learning_rate * dW1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    DEBUG = args.debug
    printd("debug mode is ON", DEBUG)

    printd("Begin construct basic MLP", DEBUG)
    # layerSchemas = [(ActivationFunction.LOGISTIC, 5), (ActivationFunction.TANH, 5),
    #                 (ActivationFunction.TANH, 5), (ActivationFunction.RELU, 5)]
    # basicMLP = MLP(5, layerSchemas)
    # basicMLP.forwardPass(np.array([1, 2, 3, 4, 5]))

    # Simple binary classifier for oranges and apples
    training_data = [(np.array([0.7, 0.8]), 1),
                     (np.array([0.2, 0.5]), 0),
                     (np.array([0.9, 0.7]), 1),
                     (np.array([0.4, 0.3]), 0),
                     (np.array([0.6, 0.6]), 1),
                     (np.array([0.3, 0.4]), 0)
                     ]

    layers = [MatrixOutMLPLayer(2, 2, ActivationFunction.RELU),
              MatrixOutMLPLayer(2, 2, ActivationFunction.RELU),
              ScalarOutMLPLayer(1, 2, ActivationFunction.SIGMOID)]
    basicMLP = MLP(layers)
    for datum in training_data:
        basicMLP.forwardPass(datum[0])
        basicMLP.backPropSingleError(datum)
