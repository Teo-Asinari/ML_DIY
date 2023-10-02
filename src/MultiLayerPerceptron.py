import argparse
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

    def initLayers(self) -> List[np.ndarray]:
        layers = [np.random.rand(self.layer_schemas[0][1], self.input_size)]

        prev_layer_size = self.layer_schemas[0][1]

        for layer in self.layer_schemas[1:]:
            layers.append(np.random.rand(layer[1], prev_layer_size))
            prev_layer_size = layer[1]

        printd("These are the layers: " + str(layers), DEBUG)
        return layers

    def forwardPass(self, input_vec: np.ndarray) -> np.ndarray:
        currVal = input_vec
        printd("curr_val start is: " + str(currVal), DEBUG)
        for layerIdx in range(len(self.layers)):
            currActivationFunc = self.layer_schemas[layerIdx][0]
            currActivation = np.dot(self.layers[layerIdx], currVal)
            printd("curr_val before activation func is: " + str(currActivation), DEBUG)
            currVal = currActivationFunc(currActivation)
            printd("curr_val after activation func is: " + str(currVal), DEBUG)
        printd("Final curr_val is: " + str(currVal), DEBUG)
        return currVal


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

    layerSchemas = [(ActivationFunction.LOGISTIC, 2), (ActivationFunction.TANH, 2),
                    (ActivationFunction.RELU, 1)]
    basicMLP = MLP(2, layerSchemas)
    for datum in training_data:
        basicMLP.forwardPass(datum[0])
