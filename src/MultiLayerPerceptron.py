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

    def __init__(self, input_size, input_layer_schema, hidden_layer_schemas, output_layer_schema):
        self.input_size = input_size
        self.input_layer_schema = input_layer_schema
        self.hidden_layer_schemas = hidden_layer_schemas
        self.output_layer_schema = output_layer_schema

    def initLayers(self):
        self.layers = []
        self.layers.append(np.random.rand((self.input_layer_schema[1], self.input_size)))

        prev_layer_size = self.input_layer_schema[1]
        self.layers.append(np.random.rand((self.hidden_layer_schemas[0][1], prev_layer_size)))
        prev_layer_size = self.hidden_layer_schemas[0][1]

        for hiddenLayer in self.hidden_layer_schemas[1:]:
            self.layers.append(np.random.rand((hiddenLayer[1], prev_layer_size)))
            prev_layer_size = hiddenLayer[1]

        self.layers.append(np.random.rand((self.output_layer_schema[1], prev_layer_size)))
        print("These are the layers: " + str(self.layers))

#    def forwardPass(self, input_vec):


if __name__ == '__main__':
    print("Begin construct basic MLP")
    inputLayerSchema = (ActivationFunction.LOGISTIC, 5)
    hiddenLayerSchemas = [(ActivationFunction.TANH, 5), (ActivationFunction.TANH, 5)]
    outputLayerSchema = (ActivationFunction.RELU, 5)
    basicMLP = MLP(5, inputLayerSchema, hiddenLayerSchemas, outputLayerSchema)
    basicMLP.initLayers()
