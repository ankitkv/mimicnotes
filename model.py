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
        with tf.variable_scope("Global"):
            self.global_step = tf.get_variable('global_step', shape=[],
                                               initializer=tf.zeros_initializer(tf.int32),
                                               trainable=False, dtype=tf.int32)
