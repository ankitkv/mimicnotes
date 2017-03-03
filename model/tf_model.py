from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model


class TFModel(model.Model):
    '''Base class for all TensorFlow models.'''

    def __init__(self, config, vocab, label_space_size):
        super(TFModel, self).__init__(config, vocab, label_space_size)
        with tf.variable_scope("Common"):
            self.global_step = tf.get_variable('global_step', shape=[],
                                               initializer=tf.zeros_initializer(dtype=tf.int32),
                                               trainable=False, dtype=tf.int32)
            self.lr = tf.get_variable("lr", shape=[],
                                     initializer=tf.constant_initializer(self.config.learning_rate),
                                     trainable=False)
        if config.l1_reg > 0.0:
            self.l1_reg = tf.contrib.layers.l1_regularizer(config.l1_reg)
        else:
            self.l1_reg = lambda x: tf.zeros([])
        if config.l2_reg > 0.0:
            self.l2_reg = tf.contrib.layers.l2_regularizer(config.l2_reg)
        else:
            self.l2_reg = lambda x: tf.zeros([])
        if config.optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)

    def initialize(self, session, load_file, verbose=True):
        '''Load a model from a saved file or initialize it if no valid load file given'''
        self.saver = tf.train.Saver(max_to_keep=None)
        if load_file:
            # try to restore a saved model file
            self.saver.restore(session, load_file)
            if verbose:
                print("Model restored from", load_file)
            # update the learning rate when a model is loaded. this should be replaced with
            # something nicer once we have learning rate decay
            session.run(tf.assign(self.lr, self.config.learning_rate))
        else:
            session.run(tf.global_variables_initializer())
            if verbose:
                print("No model file to load, new model initialized.")

    def save(self, session, save_file, overwrite, verbose=True):
        '''Save model to file'''
        global_step = None
        if not overwrite:
            global_step = self.global_step
        if verbose:
            print("Saving model...")
        save_file = self.saver.save(session, save_file, global_step=global_step)
        if verbose:
            print("Saved to", save_file)

    def minimize_loss(self, loss):
        '''Returns a train op that minimizes given loss'''
        return self.optimizer.minimize(loss, global_step=self.global_step)
