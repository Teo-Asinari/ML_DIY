import argparse

import numpy as np

from src.models.MultiLayerPerceptron import BasicMLP
from src.utils.DebugUtils import printd

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
