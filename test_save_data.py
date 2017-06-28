#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - simba

import numpy as np
from utils.data_utils import DataMaker as dm

LMDB_PATH = './train_lmdb'
TFRECODER_PATH = './train.tfrecords'

train_x = np.ones((100, 3, 100, 100), dtype=np.float32)
train_y = np.zeros((100,), dtype=np.int)

dm.save_lmdb(LMDB_PATH, train_x, train_y)
print 'Save %s Done' % LMDB_PATH

dm.save_tf_recoders(TFRECODER_PATH, train_x, train_y)
print 'Save %s Done' % TFRECODER_PATH
