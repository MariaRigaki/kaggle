#!/usr/bin/env python
# coding: utf-8

"""
prediction code for classification, using batches

Based on the work of Zygmunt Zając
see http://fastml.com/how-to-get-predictions-from-pylearn2/
"""

import sys
import numpy as np

import pandas as pd

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

if __name__ == "__main__":
    try:
        model_path = sys.argv[1]
        test_path = sys.argv[2]
        out_path = sys.argv[3]
    except IndexError:
        print "Usage: predict.py <model file> <test file> <output file>"
        print "       predict.py saved_model.pkl test_x.csv predictions.csv\n"
        quit(-1)

    print "loading model..."

    try:
        model = serial.load(model_path)
    except Exception, e:
        print "error loading {}:".format(model_path)
        print e
        quit(-1)

    print "setting up symbolic expressions..."

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)

    # Get both the probability and the class
    Y = T.max_and_argmax(Y, axis=1)
    f = function([X], Y)

    print "loading data and predicting..."

    # Use pandas to read the CSV
    df = pd.read_csv(test_path)
    x = df.values[:, 1:]

    # We need to predict in batches because the test set is quite big
    batch_size = 10000

    p_result = []
    y_result = []
    for i in range(x.shape[0]/batch_size):
        temp = []
        p, y = f(np.asarray(x[i*batch_size:(i+1)*batch_size, :], dtype=np.float32))
        y_result.extend(y)
        p_result.extend(p)

    # Make sure we get the correct number of outputs
    # print len(p_result)

    # Create the rank order
    rank_order = np.argsort(p_result) + 1
    #rank_order = list(sorted_probs)
    #for val, idx in zip(range(1, len(sorted_probs) + 1), sorted_probs):
    #    sorted_probs[idx] = val

    f2 = lambda ii: 's' if ii == 1 else 'b'
    yy = [f2(l) for l in y_result]

    print "writing predictions..."

    res_df = pd.DataFrame({"EventId": df.EventId, "RankOrder": rank_order, "Class": yy})
    res_df.to_csv(out_path, index=False, columns=["EventId", "RankOrder", "Class"])


