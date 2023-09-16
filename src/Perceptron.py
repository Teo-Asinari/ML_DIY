import numpy as np
from numpy.typing import ArrayLike

testData = [(np.array([1, 2]), 1),
            (np.array([0, 0]), 0),
            (np.array([1, 0.5]), 0),
            (np.array([2, 3]), 1),
            (np.array([2, 1]), 0)]


class Perceptron:
    weight = np.arange(3)
    learning_rate = 0.3

    def eval(self, x):
        funct_val = np.dot(self.weight, x)
        cond = funct_val > 0
        return 1 if cond else 0

    def train(self, data):
        for datum in data:
            fire_result = self.eval(datum[0])
            print("Updating weight")
            print("Current weights: " + str(self.weight))
            update = self.learning_rate * (datum[1] - fire_result) * datum[0]
            print("Update: " + str(update))
            self.weight = self.weight + update
            print("Updated weight")
            print("Current weights: " + str(self.weight))

    def calculate_loss(self, data):
        loss = 0
        for datum in data:
            loss += abs(datum[1] - self.eval(datum[0]))
        loss = loss / len(data)
        return loss

    def __init__(self, weight: ArrayLike, learning_rate):
        self.weight = weight
        self.learning_rate = learning_rate


if __name__ == '__main__':
    perceptron = Perceptron(np.array([1, 1, 1]), 0.1)
    firing_result = str(perceptron.eval(np.array([-1, 2, 0])))
    print("Basic firing test: " + firing_result)

    print("===================")
    print("Basic Training Test")
    print("===================")
    train_perceptron = Perceptron(np.array([0.001, 0.001]), 0.1)
    print("Weights before training: " + str(train_perceptron.weight))
    print("Loss before training: " + str(train_perceptron.calculate_loss(testData)))
    train_perceptron.train(testData)
    print("Weights after training: " + str(train_perceptron.weight))
    print("Loss after training: " + str(train_perceptron.calculate_loss(testData)))
