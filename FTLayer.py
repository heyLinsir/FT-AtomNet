import sys, os, random, math

import tensorflow as tf
import numpy as np

from utils import *

class FTLayer(object):
    """docstring for FTLayer."""
    def __init__(self, config):
        super(FTLayer, self).__init__()
        self.config = config

    def calc_amplitude(self, random_frequency, shape_list=[784, 28, 1], activation_fn=tf.tanh):
        random_frequency = tf.tile(tf.expand_dims(random_frequency, axis=0), [self.config.num_class, 1, 1])
        for i, (in_dim, out_dim) in enumerate(zip(shape_list[:-1], shape_list[1:])):
            W = weight_variable([self.config.num_class, in_dim, out_dim], name='fc_W1')
            bias = weight_variable([self.config.num_class, 1, out_dim], name='fc_bias1')
            random_frequency = tf.einsum('aij,ajk->aik', random_frequency, W) + tf.tile(bias, [1, random_frequency.get_shape()[1], 1])
            if i < len(shape_list) - 2:
                random_frequency = activation_fn(random_frequency)
        return tf.reshape(random_frequency, [self.config.num_class, -1]) # num_class * sample_number

    def calc_sin(self, x, random_frequency):
        return tf.sin(tf.matmul(random_frequency, tf.transpose(x, [1, 0]))) # sample_number * batch_size

    def forward(self, x, random_frequency):
        """
        x: batch_size * input_dim
        random_frequency: sample_number * input_dim
        """
        amplitude = tf.tile(tf.expand_dims(self.calc_amplitude(random_frequency), axis=2), [1, 1, self.config.batch_size]) # num_class * sample_number * batch_size
        sin_value = tf.tile(tf.expand_dims(self.calc_sin(x, random_frequency), axis=0), [self.config.num_class, 1, 1]) # num_class * sample_number * batch_size
        expection = tf.transpose(tf.reduce_mean(amplitude * sin_value, axis=1), [1, 0]) # batch_size * num_class
        prob = tf.nn.softmax(expection)
        return expection, prob
