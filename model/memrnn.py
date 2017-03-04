from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model
import util


class MemoryRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh):
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "memory_lstm_cell"):
            c, h = state
            concat = util.linear([inputs, h], (3 * self._num_units))
            j, f, o = tf.split(value=concat, num_or_size_splits=3, axis=1)
            forget = tf.sigmoid(f)
            new_c = forget * c + (1. - forget) * self._activation(j)
            new_h = self._activation(new_c) * tf.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            return new_h, new_state


class MemoryRNNModel(model.TFModel):
    '''The memory LSTM model.'''

    def __init__(self, config, vocab, label_space_size):
        super(MemoryRNNModel, self).__init__(config, vocab, label_space_size)
        self.notes = tf.placeholder(tf.int32, [config.batch_size, None], name='notes')
        self.lengths = tf.placeholder(tf.int32, [config.batch_size], name='lengths')
        self.labels = tf.placeholder(tf.float32, [config.batch_size, label_space_size],
                                     name='labels')
        with tf.device('/cpu:0'):
            init_width = 0.5 / config.word_emb_size
            self.embeddings = tf.get_variable('embeddings', [len(vocab.vocab),
                                                             config.word_emb_size],
                                              initializer=tf.random_uniform_initializer(-init_width,
                                                                                        init_width),
                                              trainable=config.train_embs)
            embed = tf.nn.embedding_lookup(self.embeddings, self.notes)

        # TODO try different activations (will need changes with loss as well)
        cell = MemoryRNNCell(label_space_size)

        # recurrence
        _, last_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=self.lengths,
                                          swap_memory=True, dtype=tf.float32)
        self.probs = (last_state.c + 1.0) / 2.0
        clipped_probs1 = tf.maximum(self.probs, 1e-8)
        clipped_probs0 = 1. - tf.minimum(self.probs, 1. - 1e-8)

        loss = self.labels * -tf.log(clipped_probs1) + (1. - self.labels) * -tf.log(clipped_probs0)
        self.loss = tf.reduce_mean(loss)
        self.train_op = self.minimize_loss(self.loss)


class MemoryRNNRunner(model.RecurrentNetworkRunner):
    '''Runner for the memory LSTM model.'''

    def __init__(self, config, session):
        super(MemoryRNNRunner, self).__init__(config, session, ModelClass=MemoryRNNModel)
