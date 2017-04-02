from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
import tensorflow as tf

import model


class PartialRNNModel(model.TFModel):
    '''The grounded RNN model where k dims of hidden state are projected to grounded dims.'''

    def __init__(self, config, vocab, label_space_size):
        super(PartialRNNModel, self).__init__(config, vocab, label_space_size)
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

        with tf.variable_scope('gru', initializer=tf.contrib.layers.xavier_initializer()):
            cell = tf.contrib.rnn.GRUCell(config.latent_size + config.hidden_size)

        # recurrence
        out, last_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=self.lengths,
                                            swap_memory=True, dtype=tf.float32)
        # concatenate 1's to the input to learn label bias
        latent = last_state[:, :config.latent_size]
        if config.grnn_summary == 'sigmoid':
            latent = tf.concat([latent, tf.ones([config.batch_size, 1])], 1)
            W = tf.get_variable('W', [config.latent_size + 1, label_space_size])
            logits = tf.matmul(latent, W)
            self.probs = tf.sigmoid(logits)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                       labels=self.labels))
        elif config.grnn_summary == 'cosine':  # doesn't work at all
            latent = tf.concat([latent, tf.ones([config.batch_size, 1])], 1)
            W = tf.get_variable('W', [config.latent_size + 1, label_space_size])
            logits = tf.matmul(latent, W)
            Xnorm = tf.sqrt(tf.reduce_sum(tf.square(latent), 1, keep_dims=True))
            Wnorm = tf.sqrt(tf.reduce_sum(tf.square(W), 0, keep_dims=True))
            logits /= tf.matmul(Xnorm, Wnorm)
            self.probs = ((logits * (1 - 2*1e-8)) + 1) / 2
            loss = self.labels * -tf.log(self.probs) + (1. - self.labels) * -tf.log(1. - self.probs)
            self.loss = tf.reduce_mean(loss)
        elif config.grnn_summary == 'softmax':
            W = tf.get_variable('W', [config.latent_size, label_space_size])
            W = tf.nn.softmax(W, 0)
            logits = tf.matmul(latent, W)
            self.probs = ((logits * (1 - 2*1e-8)) + 1) / 2
            loss = self.labels * -tf.log(self.probs) + (1. - self.labels) * -tf.log(1. - self.probs)
            self.loss = tf.reduce_mean(loss)
        elif config.grnn_summary == 'fixed':
            Wm = np.zeros([config.latent_size, label_space_size], dtype=np.float32)
            for i in xrange(label_space_size):
                indices = np.random.randint(0, config.latent_size, size=[config.grnn_fixedsize])
                Wm[:,i][indices] = 1.0
            W = tf.get_variable('W', [config.latent_size, label_space_size]) * Wm
            logits = tf.matmul(latent, W)
            self.probs = tf.sigmoid(logits)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                       labels=self.labels))

        self.train_op = self.minimize_loss(self.loss)


class PartialRNNRunner(model.RecurrentNetworkRunner):
    '''Runner for the partial RNN model.'''

    def __init__(self, config, session):
        super(PartialRNNRunner, self).__init__(config, session, ModelClass=PartialRNNModel)
