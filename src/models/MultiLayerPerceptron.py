import argparse

import numpy as np

from ..utils.DebugUtils import printd
from ..utils.ActivationFunction import activationFunc, ActivationFunction

from typing import List


DEBUG = False
class MLPLayer:
    def __init__(self, inputsize: int = 2, size: int = 2, func: activationFunc =
    ActivationFunction.RELU, downFactor = 2):
        self.inputsize = inputsize
        self.size = size
        self.func = func
        self.weights = np.random.rand(size, inputsize) * 10 ** -downFactor
        self.activation = np.zeros((size, 1))
        self.output = np.zeros((size,1))

    def __repr__(self):
        return f"MLPLayer(inputsize={self.inputsize}, " \
               f"size={self.size}, " \
               f"func={self.func}, " \
               f"weights={self.weights.tolist()}, " \
               f"activation={self.activation}, " \
               f"output={self.output})"


class MLP:
    def __init__(self, layers: List[MLPLayer] = [], learning_rate: float  = 0):
        self.layers = layers
        self.learning_rate = learning_rate

    def forwardPass(self, input_vec: np.ndarray):
        currVal = input_vec
        for layer in self.layers:
            currActivationFunc = layer.func
            currActivation = np.dot(layer.weights, currVal)
            layer.activation = currActivation
            currVal = currActivationFunc(currActivation)
            layer.output = currVal
        return currVal

# Hard-code the backprop matrices (derived by hand) for a small MLP of known layout
class BasicMLP(MLP):
    def __init__(self):
        self.layers = [MLPLayer(2, 2, ActivationFunction.RELU),
         MLPLayer(2, 2, ActivationFunction.RELU),
         MLPLayer(2, 1, ActivationFunction.SIGMOID)]
        # self.learning_rate = 0.45
        self.learning_rate = 0.1

    def backPropSingleError(self, datum):
       input = datum[0]
       expected = datum[1]
       actual = self.forwardPass(input)

       dError = 1
       dOutput2 = expected - actual
       dLayer2 = dOutput2 * ActivationFunction.SIGMOID_DERIV(self.layers[2].activation)
       dW2 = dLayer2 * np.transpose(self.layers[1].output)
       dOutput1 = dLayer2 * np.transpose(self.layers[2].weights)
       dLayer1 = np.multiply(dOutput1, ActivationFunction.RELU_DERIV(self.layers[1].activation)) # hadamard/elementwise product
       dW1 = np.matmul(dLayer1, np.transpose(self.layers[0].output))
       dOutput0 = np.matmul(np.transpose(self.layers[1].weights), dLayer1)
       dLayer0 = np.multiply(dOutput0, ActivationFunction.RELU_DERIV(self.layers[0].activation))
       dW0 = np.matmul(dLayer0, np.transpose(input))

       self.layers[2].weights += self.learning_rate * dW2
       self.layers[1].weights += self.learning_rate * dW1
       self.layers[0].weights += self.learning_rate * dW0

       print("Backprop single error. Expected: " + str(expected) +
              ", Actual: " + str(actual)
              + ", dOutput2: "  + str(dOutput2)
              + ", dW2: " + str(dW2)
              + ", dW1: " + str(dW1)
              + ", dW0: " + str(dW0))

    def getMSEForDataSet(self, dataSet):
        squaredError = 0
        for datum in dataSet:
            actual = self.forwardPass(datum[0])
            expected = datum[1]
            squaredError += pow(expected - actual, 2)
        mse = squaredError / len(dataSet)
        print("MSE is: " + str(mse))

    def trainingRun(self, training_data):
        self.getMSEForDataSet(training_data)
        for datum in training_data:
            self.backPropSingleError(datum)
        self.getMSEForDataSet(training_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    DEBUG = args.debug
    printd("debug mode is ON", DEBUG)

    printd("Begin construct basic MLP", DEBUG)

    # Simple binary classifier for oranges and apples
    training_data = [(np.array([[0.7], [0.8]]), 1),
                     (np.array([[0.2], [0.5]]), 0),
                     (np.array([[0.9], [0.7]]), 1),
                     (np.array([[0.4], [0.3]]), 0),
                     (np.array([[0.6], [0.6]]), 1),
                     (np.array([[0.3], [0.4]]), 0)
                     ]

    basicMLP = BasicMLP()

    basicMLP.trainingRun(training_data)
