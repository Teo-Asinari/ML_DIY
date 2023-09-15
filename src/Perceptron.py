import numpy as np


class Perceptron:
    weight = np.arange(3)
    bias = np.arange(3)

    def eval(self, x):
        funct_val = np.dot(self.weight, x) + self.bias
        cond = np.any(funct_val > 0)
        return 1 if cond else 0

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias


if __name__ == '__main__':
    perceptron = Perceptron(np.array([1, 1, 1]), np.array([1, 1, 1]))
    firing_result = str(perceptron.eval(np.array([-1, -1, -1])))
    print("Basic firing test: " + firing_result)
