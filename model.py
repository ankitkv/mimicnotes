from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Model(object):
    '''Base class for all models.'''

    def __init__(self, config, vocab, label_space_size):
        self.config = config
        self.vocab = vocab
        self.label_space_size = label_space_size
        with tf.variable_scope("Common"):
            self.global_step = tf.get_variable('global_step', shape=[],
                                               initializer=tf.zeros_initializer(tf.int32),
                                               trainable=False, dtype=tf.int32)
            self.lr = tf.get_variable("lr", shape=[],
                                     initializer=tf.constant_initializer(self.config.learning_rate),
                                     trainable=False)
        if config.optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)

    def minimize_loss(self, loss):
        '''Returns a train op that minimizes given loss'''
        return self.optimizer.minimize(loss, global_step=self.global_step)
