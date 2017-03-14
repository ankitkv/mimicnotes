from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model


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
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(label_space_size + config.hidden_size)

        # recurrence
        out, _ = tf.nn.dynamic_rnn(cell, embed, sequence_length=self.lengths, swap_memory=True,
                                   dtype=tf.float32)
        self.step_probs = ((out[:, :, :label_space_size] * (1 - 2*1e-6)) + 1) / 2
        self.probs = ((out[:, -1, :label_space_size] * (1 - 2*1e-6)) + 1) / 2
        loss = self.labels * -tf.log(self.probs) + (1. - self.labels) * -tf.log(1. - self.probs)
        self.loss = tf.reduce_mean(loss)
        self.train_op = self.minimize_loss(self.loss)


class NormalizedLSTMRunner(model.MemoryRNNRunner):
    '''Runner for the normalized LSTM model.'''

    def __init__(self, config, session):
        super(NormalizedLSTMRunner, self).__init__(config, session, ModelClass=NormalizedLSTMModel)
