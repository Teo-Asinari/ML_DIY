import numpy as np
from numpy.random import default_rng
import random


rng = default_rng(12345)


class Perceptron():
    def __init__(self, dim=2, learning_rate=0.5):
        self.w = rng.integers(low=0, high=100, size=dim)/100
        self.b = random.randint(0, 100)/100
        self.learning_rate = learning_rate

    def thresholdFunction(self, x):
        return 1 if np.dot(self.w, x) + self.b > 0 else 0

    def learningProc(self, learning_samples):
        for learning_sample in learning_samples:
            output = self.thresholdFunction(learning_sample[0])
            learning_update = self.learning_rate * np.multiply(np.subtract(learning_sample[1], output), learning_sample[0])
            print("learning update for: " + str(learning_sample) + " is: " + str(learning_update))
            self.w = np.add(self.w, learning_update)

#  def deltaRuleLearningProc(self):
    def __str__(self):
        return "weights: " + np.array2string(self.w) + " bias: " + str(self.b)\
                + " learning_rate: " + str(self.learning_rate)


if __name__ == '__main__':
    learning_samples = [[np.array([1, 1]), 1], [np.array([0, 0]), 0]]
    testPerceptron = Perceptron(2)
    print(testPerceptron)
    print("sample shape: " + str(learning_samples[0][0].shape))
    print("weights shape: " + str(testPerceptron.w.shape))
    print("Output for [[1,1],1]: " + str(testPerceptron.thresholdFunction(learning_samples[0][0])))
    print("Output for [[0,0],0]: " + str(testPerceptron.thresholdFunction(learning_samples[1][0])))
    testPerceptron.learningProc(learning_samples)
    print(testPerceptron)
