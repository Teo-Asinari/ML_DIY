import argparse

import numpy as np

from Utils.DebugUtils import printd
from Utils.ActivationFunction import activationFunc, ActivationFunction

from typing import List


DEBUG = False
class MLPLayer:
    def __init__(self, inputsize: int = 2, size: int = 2, func: activationFunc =
    ActivationFunction.RELU):
        self.inputsize = inputsize
        self.size = size
        self.func = func
        self.weights = np.random.rand(size, inputsize)
        self.activation = None
        self.output = None

    def __repr__(self):
        return f"MLPLayer(inputsize={self.inputsize}, " \
               f"size={self.size}, " \
               f"func={self.func}, " \
               f"weights={self.weights.tolist()}, " \
               f"activation={self.activation}, " \
               f"output={self.output})"

class MatrixOutMLPLayer(MLPLayer):
    def __init__(self, inputsize: int = 2, size: int = 2, func: activationFunc =
    ActivationFunction.RELU):
        super(MatrixOutMLPLayer, self).__init__(inputsize, size, func)
        self.activation = np.zeros((size, 1))
        self.output = np.zeros((size, 1))

class ScalarOutMLPLayer(MLPLayer):
    def __init__(self, inputsize : int = 2, size: int = 2, func: activationFunc = ActivationFunction.RELU):
        super(ScalarOutMLPLayer, self).__init__(inputsize, size, func)
        self.activation = 0
        self.output = 0

class MLP:
    def __init__(self, layers: List[MLPLayer] = [], learning_rate: float  = 0):
        self.layers = layers
        self.learning_rate = learning_rate

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
        self.layers = [MatrixOutMLPLayer(2, 2, ActivationFunction.RELU),
         MatrixOutMLPLayer(2, 2, ActivationFunction.RELU),
         ScalarOutMLPLayer(1, 2, ActivationFunction.SIGMOID)]
        self.learning_rate = 0.1

    def backPropSingleError(self, datum):
       input = datum[0]
       expected = datum[1]
       actual = self.forwardPass(input)
       error = 0.5 * pow(expected - actual, 2)

       printd("Starting backprop. Error is " + str(error), DEBUG)

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

       self.layers[2].weights -= self.learning_rate * dW3
       self.layers[1].weights -= self.learning_rate * dW2
       self.layers[0].weights -= self.learning_rate * dW1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    DEBUG = args.debug
    printd("debug mode is ON", DEBUG)

    printd("Begin construct basic MLP", DEBUG)

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

