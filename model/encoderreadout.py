from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

import model


class GRNNCell(tf.contrib.rnn.RNNCell):
    """GRNN cell."""

    def __init__(self, label_space_size, activation=tf.tanh, variables={}):
        self._label_space_size = label_space_size
        self._activation = activation
        self._variables = variables

    @property
    def state_size(self):
        return self._label_space_size

    @property
    def output_size(self):
        return self._label_space_size

    def diagonal_linear(self, inputs, var_scope):
        """Similar to linear, but with the weight matrix restricted to be partially diagonal."""
        diagonal = self._variables[var_scope]['Diagonal']
        right_matrix = self._variables[var_scope]['RightMatrix']
        bias = self._variables[var_scope]['Bias']
        diag_res = inputs[:, :self._label_space_size] * tf.expand_dims(diagonal, 0)
        res = tf.matmul(inputs[:, self._label_space_size:], right_matrix) + diag_res
        return tf.nn.bias_add(res, bias)

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or "diagonal_gru_cell"):
            with tf.variable_scope("r_gate"):
                r = tf.sigmoid(self.diagonal_linear(tf.concat([state, inputs], 1), 'r_gate'))
            with tf.variable_scope("u_gate"):
                u = tf.sigmoid(self.diagonal_linear(tf.concat([state, inputs], 1), 'u_gate'))
            with tf.variable_scope("candidate"):
                c = self._activation(self.diagonal_linear(tf.concat([r * state, inputs], 1),
                                                          'candidate'))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class Encoder(object):
    '''Takes word embeddings and encodes them to a representation to read out from.'''

    def encode(self, inputs, lengths):
        raise NotImplementedError


class ReadOut(object):
    '''Take encoder representations and produce classifications.'''

    def classify(self, inputs, lengths):
        raise NotImplementedError


class RecurrentEncoder(Encoder):

    def __init__(self, hidden_size, reconcat_input):
        self.hidden_size = hidden_size
        self.reconcat_input = reconcat_input

    def encode(self, inputs, lengths):
        with tf.variable_scope('gru', initializer=tf.contrib.layers.xavier_initializer()):
            cell = tf.contrib.rnn.GRUCell(self.hidden_size)
            out, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=lengths, swap_memory=True,
                                       dtype=tf.float32)
        if self.reconcat_input:
            out = tf.concat([inputs, out], 2)
        return out, lengths


class ConvolutionalEncoder(Encoder):

    def __init__(self):
        pass  # TODO

    def encode(self, inputs, lengths):
        pass  # TODO


class EmbeddingEncoder(Encoder):

    def encode(self, inputs, lengths):
        return inputs, lengths


class GroundedReadOut(ReadOut):

    def __init__(self, label_space_size):
        self.label_space_size = label_space_size

    def classify(self, inputs, lengths):
        with tf.variable_scope('grnn', initializer=tf.contrib.layers.xavier_initializer()):
            variables = collections.defaultdict(dict)
            for sc_name, bias_start in [('r_gate', 1.0), ('u_gate', 1.0), ('candidate', 0.0)]:
                with tf.variable_scope('rnn/diagonal_gru_cell/' + sc_name):
                    diagonal = tf.get_variable("Diagonal", [self.label_space_size],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                    nondiag_size = inputs.get_shape()[2].value
                    right_matrix = tf.get_variable("RightMatrix", [nondiag_size,
                                                                   self.label_space_size],
                                                   dtype=tf.float32)

                    bias_term = tf.get_variable("Bias", [self.label_space_size], dtype=tf.float32,
                                                initializer=tf.constant_initializer(bias_start))

                variables[sc_name]['Diagonal'] = diagonal
                variables[sc_name]['RightMatrix'] = right_matrix
                variables[sc_name]['Bias'] = bias_term

            cell = GRNNCell(self.label_space_size, variables=variables)

            # forward recurrence
            out, last_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=lengths,
                                                swap_memory=True, dtype=tf.float32)

        probs = (last_state[:, :self.label_space_size] + 1) / 2
        step_probs = (out[:, :, :self.label_space_size] + 1) / 2
        label_bias = tf.get_variable('label_bias', [self.label_space_size],
                                     initializer=tf.constant_initializer(0.0))
        # y = sigmoid(inverse_sigmoid(y) + b)
        exp_bias = tf.expand_dims(tf.exp(-label_bias), 0)
        probs = probs / (probs + ((1 - probs) * exp_bias))
        exp_bias = tf.expand_dims(exp_bias, 0)
        step_probs = step_probs / (step_probs + ((1 - step_probs) * exp_bias))

        return probs, step_probs, False


class MaxReadOut(ReadOut):

    def __init__(self, label_space_size):
        self.label_space_size = label_space_size

    def pool(self, preds):
        return tf.reduce_max(preds, 1)

    def classify(self, inputs, lengths):
        with tf.variable_scope('pool', initializer=tf.contrib.layers.xavier_initializer()):
            in_shape = tf.shape(inputs)
            inputs = tf.reshape(inputs, [-1, inputs.get_shape()[2].value])
            out = tf.layers.dense(inputs, self.label_space_size)
            out = tf.reshape(out, tf.concat([in_shape[:2], [self.label_space_size]], 0))
            # TODO max/mean only within the lengths
            return self.pool(out), None, True


class MeanReadOut(MaxReadOut):

    def __init__(self, label_space_size):
        super(MeanReadOut, self).__init__(label_space_size)

    def pool(self, preds):
        return tf.reduce_mean(preds, 1)


class EncoderReadOutModel(model.TFModel):
    '''The encoder-readout model.'''

    def __init__(self, config, vocab, label_space_size, scope=None):
        super(EncoderReadOutModel, self).__init__(config, vocab, label_space_size, scope=scope)
        self.lengths = tf.placeholder(tf.int32, [config.batch_size], name='lengths')
        self.labels = tf.placeholder(tf.float32, [config.batch_size, label_space_size],
                                     name='labels')

        with tf.device('/cpu:0'):
            self.notes = tf.placeholder(tf.int32, [config.batch_size, None], name='notes')
            init_width = 0.5 / config.word_emb_size
            self.embeddings = tf.get_variable('embeddings', [len(vocab.vocab),
                                                             config.word_emb_size],
                                              initializer=tf.random_uniform_initializer(-init_width,
                                                                                        init_width),
                                              trainable=config.train_embs)
            inputs = tf.nn.embedding_lookup(self.embeddings, self.notes)

        if config.encoder == 'gru':
            encoder = RecurrentEncoder(config.hidden_size, config.reconcat_input)
        elif config.encoder == 'conv':
            encoder = ConvolutionalEncoder()
        elif config.encoder == 'embs':
            encoder = EmbeddingEncoder()

        encoded, lengths = encoder.encode(inputs, self.lengths)

        if config.readout == 'grnn':
            readout = GroundedReadOut(label_space_size)
        elif config.readout == 'max':
            readout = MaxReadOut(label_space_size)
        elif config.readout == 'mean':
            readout = MeanReadOut(label_space_size)

        self.probs, self.step_probs, logprobs = readout.classify(encoded, lengths)

        if logprobs:
            loss = tf.losses.sigmoid_cross_entropy(self.labels, self.probs,
                                                   reduction=tf.losses.Reduction.NONE)
            self.probs = tf.sigmoid(self.probs)
            if self.step_probs is not None:
                self.step_probs = tf.sigmoid(self.probs)
        else:
            # fix potential numerical instability
            self.probs = self.probs * (1 - 2*1e-6) + 1e-6
            if self.step_probs is not None:
                self.step_probs = self.step_probs * (1 - 2*1e-6) + 1e-6
            loss = self.labels * -tf.log(self.probs) + (1. - self.labels) * -tf.log(1. - self.probs)

        self.loss = tf.reduce_mean(loss)
        self.train_op = self.minimize_loss(self.loss)


class EncoderReadOutRunner(model.RecurrentNetworkRunner):
    '''Runner for the encoder-readout model.'''

    def __init__(self, config, session, verbose=True):
        super(EncoderReadOutRunner, self).__init__(config, session, ModelClass=EncoderReadOutModel)
