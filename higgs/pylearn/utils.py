__author__ = 'marik0'

# We'll need numpy to manage arrays of data
import numpy as np
import pandas as pd

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


def load_data(start, stop):
    """
    Loads the higgs boson dataset from the Kaggle competition

    The dataset contains 250000 examples, including a classification label.

    Parameters
    ----------
    start: int
    stop: int

    Returns
    -------

    dataset : DenseDesignMatrix
        A dataset include examples start (inclusive) through stop (exclusive).
        The start and stop parameters are useful for splitting the data into
        train, validation, and test data.
    """
    df = pd.read_csv('/home/marik0/repos/kaggle/higgs/data/training.csv', index_col='EventId')

    labels = df['Label']
    f = lambda x: 1 if x == 's' else -1
    y = labels.map(f)
    y = y.reshape(y.shape[0], 1)
    #print y.shape

    one_hot = np.zeros((y.shape[0], 2))

    for i in xrange(y.shape[0]):
        label = y[i]
        if label == 1:
            one_hot[i, 1] = 1
        else:
            one_hot[i, 0] = 1


    weights = df['Weight']
    X = df.drop(['Label', 'Weight'], axis=1)

    del df

    y = np.asarray(one_hot)
    y = y.reshape(y.shape[0], 2)

    X = X.values[start:stop, :]
    y = y[start:stop, :]

    return DenseDesignMatrix(X=X, y=y)
    #return X, y