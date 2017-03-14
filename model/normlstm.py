from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model
import util


class NormalizedLSTMModel(model.TFModel):
    '''The normalized LSTM model.'''

    def __init__(self, config, vocab, label_space_size):
        super(NormalizedLSTMModel, self).__init__(config, vocab, label_space_size)
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

        with tf.variable_scope('lstm', initializer=tf.contrib.layers.xavier_initializer()):
            if config.normlstm_mem:
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(label_space_size + config.hidden_size)
            else:
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(config.hidden_size)

        # recurrence
        _, last_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=self.lengths,
                                          swap_memory=True, dtype=tf.float32)
        if config.lstm_hidden == 'c':
            last_state = last_state.c
        elif config.lstm_hidden == 'h':
            last_state = last_state.h
        if config.normlstm_mem:
            state = last_state[:, :label_space_size]
            multipliers = tf.get_variable('mult', [1, label_space_size],
                                          initializer=tf.ones_initializer())
            bias = tf.get_variable('bias', [label_space_size],
                                   initializer=tf.zeros_initializer())
            logits = tf.nn.bias_add(state * multipliers, bias)
        else:
            logits = util.linear(last_state, label_space_size)
        self.probs = tf.sigmoid(logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                   labels=self.labels))
        self.train_op = self.minimize_loss(self.loss)


class NormalizedLSTMRunner(model.MemoryRNNRunner):
    '''Runner for the normalized LSTM model.'''

    def __init__(self, config, session):
        super(NormalizedLSTMRunner, self).__init__(config, session, ModelClass=NormalizedLSTMModel)
