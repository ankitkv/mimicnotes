from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model
import util


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

        cell = tf.contrib.rnn.GRUCell(label_space_size + (config.word_emb_size * config.num_blocks))

        # recurrence
        _, last_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=self.lengths,
                                          swap_memory=True, dtype=tf.float32)
        self.probs = (last_state[:, :label_space_size] + 1.0) / 2.0
        clipped_probs1 = tf.maximum(self.probs, 1e-8)
        clipped_probs0 = 1. - tf.minimum(self.probs, 1. - 1e-8)

        loss = self.labels * -tf.log(clipped_probs1) + (1. - self.labels) * -tf.log(clipped_probs0)
        self.loss = tf.reduce_mean(loss)
        self.train_op = self.minimize_loss(self.loss)


class MemoryRNNRunner(model.RecurrentNetworkRunner):
    '''Runner for the memory LSTM model.'''

    def __init__(self, config, session):
        super(MemoryRNNRunner, self).__init__(config, session, ModelClass=MemoryRNNModel)
