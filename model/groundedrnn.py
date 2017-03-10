from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model


class GroundedRNNModel(model.TFModel):
    '''The grounded RNN model.'''

    def __init__(self, config, vocab, label_space_size):
        super(GroundedRNNModel, self).__init__(config, vocab, label_space_size)
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
        latent = tf.concat([last_state[:, :config.latent_size], tf.ones([config.batch_size, 1])], 1)
        W = tf.get_variable('W', [config.latent_size + 1, label_space_size])
        logits = tf.matmul(latent, W)
        if config.grnn_sigmoid:
            self.probs = tf.sigmoid(logits)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                       labels=self.labels))
        else:
            Xnorm = tf.sqrt(tf.reduce_sum(tf.square(latent), 1, keep_dims=True))
            Wnorm = tf.sqrt(tf.reduce_sum(tf.square(W), 0, keep_dims=True))
            logits /= tf.matmul(Xnorm, Wnorm)
            self.probs = ((logits * (1 - 2*1e-8)) + 1) / 2
            loss = self.labels * -tf.log(self.probs) + (1. - self.labels) * -tf.log(1. - self.probs)
            self.loss = tf.reduce_mean(loss)

        self.train_op = self.minimize_loss(self.loss)


class GroundedRNNRunner(model.RecurrentNetworkRunner):
    '''Runner for the grounded RNN model.'''

    def __init__(self, config, session):
        super(GroundedRNNRunner, self).__init__(config, session, ModelClass=GroundedRNNModel)
