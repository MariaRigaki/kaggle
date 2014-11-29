__author__ = 'marik0'

import pandas as pd
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


def load_data(start, stop):
    train = pd.read_csv('data/training.csv')

    labels = train[['Ca', 'P', 'pH', 'SOC', 'Sand']].values

    train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)

    co2_bands = ['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97', 'm2372.04', 'm2370.11',
                 'm2368.18', 'm2366.26', 'm2364.33', 'm2362.4',  'm2360.47', 'm2358.54',
                 'm2356.61', 'm2354.68', 'm2352.76']
    train.drop(co2_bands, axis=1, inplace=True)


    X = np.array(train)[start:stop, :3578]
    y = labels[start:stop, 4]
    y = y.reshape(y.shape[0], 1)

    return DenseDesignMatrix(X=X, y=y)


