import argparse
import os.path

import pandas as pd
import numpy as np


from src.models.MultiLayerPerceptron import BasicMLP
from src.utils.DebugUtils import printd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("Pandas version:")
    print(pd.__version__)
    DEBUG = args.debug
    printd("debug mode is ON", DEBUG)

    printd("Begin construct basic MLP", DEBUG)

    relative_path = os.path.join('..', '..', '..', 'data', 'iris2d.csv')
    df = pd.read_csv(relative_path)
    binary_mapping = {'Setosa': 1, 'Versicolor' : 0}
    df['variety'] = df['variety'].map(binary_mapping)
    input_features = df.drop(columns=['variety'])

    # training_data = list(zip(input_features.to_numpy(), df['variety']))
    training_data = list(zip(np.reshape(input_features.to_numpy(), (input_features.shape[0],
                                                                    input_features.shape[1],
                                                                    1)),
                             df['variety']))
    # training_data = list(zip(input_features.apply(lambda x: x.to_numpy().reshape(-1, 1), axis=0).values, df['variety']))

    basicMLP = BasicMLP()

    basicMLP.trainingRun(training_data)