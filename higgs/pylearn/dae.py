__author__ = 'marik0'
import os

from pylearn2.config import yaml_parse

with open(os.path.join('spdae_l1.yaml'), 'r') as f:
    l1_train = f.read()

print l1_train

l1_train = yaml_parse.load(l1_train)
l1_train.main_loop()
#
# with open(os.path.join('dae_l2.yaml'), 'r') as f:
#     l2_train = f.read()
#
# print l2_train
#
# l2_train = yaml_parse.load(l2_train)
# l2_train.main_loop()

# with open(os.path.join('dae_l3.yaml'), 'r') as f:
#     l3_train = f.read()
#
# print l3_train
#
# l3_train = yaml_parse.load(l3_train)
# l3_train.main_loop()
#
# with open(os.path.join('dae_mlp.yaml'), 'r') as f:
#     train = f.read()
#
# print train
#
# train = yaml_parse.load(train)
# train.main_loop()