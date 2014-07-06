__author__ = 'marik0'
import os
import pylearn2
import cPickle as pickle
from pylearn2.config import yaml_parse

#model = pickle.load(open('mlp_best3.pkl', 'rb'))

with open(os.path.join('mlp.yaml'), 'r') as f:
    train = f.read()

print train

train = yaml_parse.load(train)
#pylearn2.monitor.push_monitor(model, 'mlp_best3_old.pkl', transfer_experience=True)

train.main_loop()

