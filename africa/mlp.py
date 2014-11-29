__author__ = 'marik0'

import os
from pylearn2.config import yaml_parse

with open(os.path.join('mlp.yaml'), 'r') as f:
    train = f.read()

print train

train = yaml_parse.load(train)

train.main_loop()
