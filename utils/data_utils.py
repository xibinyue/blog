#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - simba

import lmdb
import caffe
import tensorflow as tf
import numpy as np
import random


class DataMaker(object):
    def __init__(self):
        super(DataMaker, self).__init__()

    @staticmethod
    def save_lmdb(dest_path, x_data, y_data):
        """save LMDB data for caffe
        :param dest_path: data path you save to
        :param x_data: X data,numpy.ndarray with shape:(N,channel,height,width)
        :param y_data: Y data(label),numpy.ndarray with shape:(N,), or list which length = N
        :return:True if save successful else False
        """
        data_env = lmdb.Environment(dest_path, map_size=int(1e12))
        assert len(x_data) == len(y_data), 'Length of data and labels must be equal!'
        try:
            with data_env.begin(write=True) as txn:
                for idx in xrange(len(x_data)):
                    datum = caffe.proto.caffe_pb2.Datum()
                    tmp = x_data[idx][None, ...]
                    datum.channels = tmp.shape[1]
                    datum.height = tmp.shape[2]
                    datum.width = tmp.shape[3]
                    datum.data = tmp.tostring()
                    datum.label = int(y_data[idx])
                    str_id = '{:05}'.format(idx)
                    txn.put(str_id, datum.SerializeToString())
            return True
        except Exception as e:
            print e.message
            return False

    @staticmethod
    def save_tf_recoders(dst_path, x_data, y_data):
        """save tfrecoder files for tensor flow, with sample shuffle
        :param dst_path: file save path
        :param x_data: X data,numpy.ndarray with shape:(N,channel,height,width)
        :param y_data: Y data(label),numpy.ndarray with shape:(N,), or list which length = N
        :return: True if save successful else False
        """
        try:
            writer = tf.python_io.TFRecordWriter(dst_path)
            seq = np.array(xrange(x_data.shape[0]), np.int32)
            random.shuffle(seq)  # shuffle data
            for idx in seq:
                data_string = x_data[idx].tobytes()
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_string])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[y_data[idx]]))
                    }
                ))
                writer.write(example.SerializeToString())
            writer.close()
            return True
        except Exception as e:
            print e.message
            return False
