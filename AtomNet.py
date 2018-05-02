import sys, os, random, math

import tensorflow as tf
import numpy as np

import FTLayer
from utils import *

class AtomNet(object):
    """docstring for AtomNet."""
    def __init__(self, config):
        super(AtomNet, self).__init__()
        self.config = config

        self.weight_for_random_frequency = np.repeat(np.asarray(range(1, 601, 1) * 5, dtype='float32').reshape([-1, 1]), self.config.input_dim, axis=1)

        self.build()

    def generate_random_frequency(self):
        return np.random.rand(self.config.num_sample, self.config.input_dim) #  * self.weight_for_random_frequency

    def build(self):
        self.random_frequency_placeholder = tf.placeholder(tf.float32, [self.config.num_sample, self.config.input_dim], 'random_frequency')
        self.x_placeholder = tf.placeholder(tf.float32, [self.config.batch_size, self.config.input_dim], 'input_x')
        self.label_placeholder = tf.placeholder(tf.uint8, [self.config.batch_size], 'label')

        one_hot_label = tf.one_hot(self.label_placeholder, depth=self.config.num_class, on_value=1., off_value=0., dtype=tf.float32)

        FT_layer = FTLayer.FTLayer(self.config)
        expection, prob = FT_layer.forward(self.x_placeholder, self.random_frequency_placeholder)

        self.loss = tf.losses.softmax_cross_entropy(one_hot_label, expection, reduction=tf.losses.Reduction.MEAN)
        self.prediction = tf.cast(tf.argmax(expection, axis=1, output_type=tf.int32), dtype=tf.uint8)
        self.right = tf.cast(tf.equal(self.prediction, self.label_placeholder), dtype=tf.float32)

        if self.config.optimizer == 'Adam':
            self.train_step = tf.train.AdamOptimizer(self.config.adam_learning_rate).minimize(self.loss)
        elif self.config.optimizer == 'SGD':
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.config.sgd_learning_rate, self.global_step, 1500, 0.96, staircase=False)
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def make_feed_dict(self, _feed_dict):
        feed_dict = {}
        feed_dict[self.random_frequency_placeholder] = self.generate_random_frequency()
        feed_dict[self.x_placeholder] = _feed_dict['x']
        feed_dict[self.label_placeholder] = _feed_dict['label']
        return feed_dict, _feed_dict['label'].shape[0]
