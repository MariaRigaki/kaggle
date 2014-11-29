__author__ = 'marik0'
#!/usr/bin/env python
# coding: utf-8

"""
prediction code for regression
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
    #Y = T.max_and_argmax(Y, axis=1)
    f = function([X], Y)

    print "loading data and predicting..."

    # Use pandas to read the CSV
    df = pd.read_csv(test_path)
    df.drop('PIDN', axis=1, inplace=True)
    co2_bands = ['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97', 'm2372.04', 'm2370.11',
             'm2368.18', 'm2366.26', 'm2364.33', 'm2362.4',  'm2360.47', 'm2358.54',
             'm2356.61', 'm2354.68', 'm2352.76']
    df.drop(co2_bands, axis=1, inplace=True)

    x = np.array(df)[:, :3578]

    y = f(x.astype(dtype=np.float32))


    # Make sure we get the correct number of outputs
    print len(y)

    print "writing predictions..."

    res_df = pd.read_csv('data/sample_submission.csv')
    #res_df['Ca'] = y
    #res_df['P'] = y
    #res_df['pH'] = y
    #res_df['SOC'] = y
    res_df['Sand'] = y
    res_df.to_csv(out_path, index=False)


