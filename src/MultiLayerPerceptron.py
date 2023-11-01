import argparse
import math

import numpy as np

from Utils.DebugUtils import printd
from Utils.ActivationFunction import activationFunc, ActivationFunction

from typing import List, Tuple

layerSchema = Tuple[activationFunc, int]

DEBUG = False

class MLP:
    def __init__(self, input_size: int, layer_schemas: List[layerSchema]):
        self.input_size = input_size
        self.layer_schemas = layer_schemas
        self.layers = self.initLayers()
        self.learning_rate = 0.1

    # Design: Each Layer is a list of [weights matrix, deriv w.r.t. loss matrix,
    # last activation, activation_func(last activation)]
    def initLayers(self) -> List[np.ndarray]:
        layers = [[np.random.rand(self.layer_schemas[0][1], self.input_size),
                   np.random.rand(self.layer_schemas[0][1], self.input_size),
                   np.zeros(self.layer_schemas[0][1], 1),
                   np.zeros(self.layer_schemas[0][1], 1)]]

        prev_layer_size = self.layer_schemas[0][1]

        for layer in self.layer_schemas[1:]:
            layers.append([np.random.rand(layer[1], prev_layer_size),
                           np.random.rand(layer[1], prev_layer_size),
                           0 if layer[1] == 1 else np.zeros(layer[1], 1),
                           0 if layer[1] == 1 else np.zeros(layer[1], 1)])
            prev_layer_size = layer[1]

        printd("These are the layers: " + str(layers), DEBUG)
        return layers

    def forwardPass(self, input_vec: np.ndarray) -> np.ndarray:
        currVal = input_vec
        printd("curr_val start is: " + str(currVal), DEBUG)
        for layerIdx in range(len(self.layers)):
            currActivationFunc = self.layer_schemas[layerIdx][0]
            currActivation = np.dot(self.layers[layerIdx][0], currVal)
            self.layers[layerIdx][2] = currActivation
            printd("curr_val before activation func is: " + str(currActivation), DEBUG)
            currVal = currActivationFunc(currActivation)
            self.layers[layerIdx][3] = currVal
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
       dLayer3 = dOutput3 * ActivationFunction.SIGMOID_DERIV(self.layers[2][2])
       dW3 = dLayer3 * np.transpose(self.layers[1][3])
       dOutput2 = dLayer3 * np.transpose(self.layers[2][0])
       dLayer2 = dOutput2 * ActivationFunction.RELU_DERIV(self.layers[0][0])
       dW2 = np.mdLayer2





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

    layerSchemas = [(ActivationFunction.RELU, 2), (ActivationFunction.RELU, 2),
                    (ActivationFunction.SIGMOID, 1), (ActivationFunction.ROUND, 1)]
    basicMLP = MLP(2, layerSchemas)
    for datum in training_data:
        basicMLP.forwardPass(datum[0])
