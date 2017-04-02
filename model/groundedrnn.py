from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six.moves import xrange

import tensorflow as tf

import model
import util

try:
    input = raw_input
except NameError:
    pass


class DiagonalGRUCell(tf.contrib.rnn.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078) with semidiagonal weights."""

    def __init__(self, label_space_size, control_size, activation=tf.tanh, positive_diag=True,
                 reuse=None):
        self._label_space_size = label_space_size
        self._control_size = control_size
        self._activation = activation
        self._positive_diag = positive_diag
        self._reuse = reuse

    @property
    def state_size(self):
        return self._label_space_size + self._control_size

    @property
    def output_size(self):
        return self._label_space_size + self._control_size

    def diagonal_linear(self, inputs, bias=True, bias_start=0.0, diag_start=0.0,
                        positive_diag=False, scope=None, initializer=None):
        """Similar to linear, but with the weight matrix restricted to be partially diagonal."""
        nondiag_size = inputs.get_shape()[1].value - self._label_space_size
        dtype = inputs.dtype
        if initializer is None:
            initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(scope or "DiagonalLinear"):
            diagonal = tf.get_variable("Diagonal", [self._label_space_size], dtype=dtype,
                                       initializer=tf.constant_initializer(diag_start, dtype=dtype))
            if positive_diag:
                diagonal = tf.nn.elu(diagonal) + 1
            diag_res = inputs[:, :self._label_space_size] * tf.expand_dims(diagonal, 0)
            if nondiag_size > 0:
                right_matrix = tf.get_variable("RightMatrix", [nondiag_size, self.state_size],
                                               dtype=dtype, initializer=initializer)
                # it's a good idea to regularize the following matrix:
                bottom_matrix = tf.get_variable("BottomMatrix",
                                                [self._label_space_size,
                                                 self.state_size - self._label_space_size],
                                                dtype=dtype, initializer=initializer)
                res = tf.matmul(inputs[:, self._label_space_size:], right_matrix)
                res += tf.concat([diag_res, tf.matmul(inputs[:, :self._label_space_size],
                                                      bottom_matrix)], 1)
            else:
                res = diag_res

            if not bias:
                return res
            bias_term = tf.get_variable("Bias", [self.state_size], dtype=dtype,
                                        initializer=tf.constant_initializer(bias_start,
                                                                            dtype=dtype))
        return tf.nn.bias_add(res, bias_term)

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or "diagonal_gru_cell", reuse=self._reuse):
            with tf.variable_scope("r_gate"):
                r = tf.sigmoid(self.diagonal_linear(tf.concat([state, inputs], 1),
                                                    True, 1.0))
            with tf.variable_scope("u_gate"):
                u = tf.sigmoid(self.diagonal_linear(tf.concat([state, inputs], 1),
                                                    True, 1.0))
            with tf.variable_scope("candidate"):
                c = self._activation(self.diagonal_linear(tf.concat([r * state, inputs], 1),
                                                          True, positive_diag=self._positive_diag))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class GroundedRNNModel(model.TFModel):
    '''The grounded RNN model.'''

    def __init__(self, config, vocab, label_space_size):
        super(GroundedRNNModel, self).__init__(config, vocab, label_space_size)
        self.lengths = tf.placeholder(tf.int32, [config.batch_size], name='lengths')
        self.labels = tf.placeholder(tf.float32, [config.batch_size, label_space_size],
                                     name='labels')

        with tf.device('/cpu:0'):
            self.notes = tf.placeholder(tf.int32, [config.batch_size, None], name='notes')
            if config.bidirectional:
                rev_notes = tf.reverse_sequence(self.notes[:, 1:], tf.maximum(self.lengths - 1, 0),
                                                seq_axis=1, batch_axis=0)
                rev_notes = tf.concat([tf.constant(vocab.eos_index,
                                       dtype=tf.int32, shape=[config.batch_size, 1]), rev_notes], 1)

            init_width = 0.5 / config.word_emb_size
            self.embeddings = tf.get_variable('embeddings', [len(vocab.vocab),
                                                             config.word_emb_size],
                                              initializer=tf.random_uniform_initializer(-init_width,
                                                                                        init_width),
                                              trainable=config.train_embs)
            embed = tf.nn.embedding_lookup(self.embeddings, self.notes)
            if config.bidirectional:
                rev_embed = tf.nn.embedding_lookup(self.embeddings, rev_notes)

        inputs = embed
        if config.bidirectional:
            with tf.variable_scope('gru_rev', initializer=tf.contrib.layers.xavier_initializer()):
                rev_cell = tf.contrib.rnn.GRUCell(config.hidden_size)
                # backward recurrence
                rev_out, _ = tf.nn.dynamic_rnn(rev_cell, rev_embed, sequence_length=self.lengths,
                                               swap_memory=True, dtype=tf.float32)
                rev_out = tf.reverse_sequence(rev_out, self.lengths, seq_axis=1, batch_axis=0)
            inputs = tf.concat([inputs, rev_out], 2)

        with tf.variable_scope('gru', initializer=tf.contrib.layers.xavier_initializer()):
            if config.diagonal_cell:
                cell = DiagonalGRUCell(label_space_size, config.hidden_size,
                                       positive_diag=config.positive_diag)
            else:
                cell = tf.contrib.rnn.GRUCell(label_space_size + config.hidden_size)
            # forward recurrence
            out, last_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.lengths,
                                                swap_memory=True, dtype=tf.float32)

        self.probs = (last_state[:, :label_space_size] + 1) / 2
        self.step_probs = (out[:, :, :label_space_size] + 1) / 2
        if config.biased_sigmoid:
            label_bias = tf.get_variable('label_bias', [label_space_size],
                                         initializer=tf.constant_initializer(0.0))
            # y = sigmoid(inverse_sigmoid(y) + b)
            exp_bias = tf.expand_dims(tf.exp(-label_bias), 0)
            self.probs = self.probs / (self.probs + ((1 - self.probs) * exp_bias))
            exp_bias = tf.expand_dims(exp_bias, 0)
            self.step_probs = self.step_probs / (self.step_probs +
                                                 ((1 - self.step_probs) * exp_bias))

        if config.grnn_loss == 'ce':
            # fix potential numerical instability
            self.probs = self.probs * (1 - 2*1e-6) + 1e-6
            self.step_probs = self.step_probs * (1 - 2*1e-6) + 1e-6
            loss = self.labels * -tf.log(self.probs) + (1. - self.labels) * -tf.log(1. - self.probs)
        elif config.grnn_loss == 'l1':
            loss = tf.abs(self.labels - self.probs)
        self.loss = tf.reduce_mean(loss)
        if self.config.l1_reg > 0.0 or self.config.l2_reg > 0.0:
            bottom_matrices = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='.*DiagonalLinear/BottomMatrix')
            accumulated = tf.concat([tf.reshape(v, [-1]) for v in bottom_matrices], 0)
            self.loss += self.l1_reg(accumulated) + self.l2_reg(accumulated)

        # optional language modeling objective for controller dims
        if config.lm_weight > 0.0:
            flat_out = tf.reshape(out[:, :-1,
                                      label_space_size:label_space_size + config.latent_size],
                                  [-1, config.latent_size])
            flat_targets = tf.reshape(self.notes[:, 1:], [-1])
            flat_mask = tf.to_float(flat_targets > 0)
            lm_logits = util.linear(flat_out, len(vocab.vocab))
            lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_targets,
                                                                     logits=lm_logits) * flat_mask
            lm_loss = tf.reduce_sum(lm_loss) / tf.maximum(tf.reduce_sum(flat_mask), 1.0)
            self.loss += config.lm_weight * lm_loss

        self.train_op = self.minimize_loss(self.loss)


class GroundedRNNRunner(model.RecurrentNetworkRunner):
    '''Runner for the grounded RNN model.'''

    def __init__(self, config, session, ModelClass=GroundedRNNModel):
        super(GroundedRNNRunner, self).__init__(config, session, ModelClass=ModelClass)

    def visualize(self, verbose=True):
        if self.config.query:
            split = self.config.query
        else:
            split = 'test'
        for batch in self.reader.get([split], force_curriculum=False):
            ops = [self.model.probs, self.model.step_probs]
            probs, step_probs = self.session.run(ops, feed_dict={self.model.notes: batch[0],
                                                                 self.model.lengths: batch[1],
                                                                 self.model.labels: batch[2]})
            for i in xrange(probs.shape[0]):
                print()
                print('=== NEW NOTE ===')
                doc_probs = step_probs[i]  # seq_len x labels
                prob = [(j, p) for j, p in enumerate(probs[i]) if p > 0.5]
                prob.sort(key=lambda x: -x[1])
                labels = collections.OrderedDict((l, True) for l, _ in prob)
                for j in xrange(len(batch[2][i])):
                    if batch[2][i, j] and j not in labels:
                        labels[j] = False
                for label, predicted in labels.items():
                    label_prob = doc_probs[:, label]  # seq_len
                    if predicted:
                        if batch[2][i, label]:
                            verdict = 'correct'
                        else:
                            verdict = 'incorrect'
                    else:
                        verdict = 'missed'
                    print()
                    print('LABEL (%s): #%d' % (verdict, label+1),
                          self.vocab.aux_names['dgn'][self.vocab.aux_vocab['dgn'][label]])
                    print('-----')
                    for k, word in enumerate(batch[0][i, :batch[1][i]]):
                        prob = label_prob[k]
                        if prob > 0.8:
                            color = util.c.OKGREEN
                        elif prob > 0.6:
                            color = util.c.WARNING
                        elif prob > 0.5:
                            color = util.c.ENDC
                        elif prob <= 0.2:
                            color = util.c.FAIL
                        elif prob <= 0.4:
                            color = util.c.HEADER
                        elif prob <= 0.5:
                            color = util.c.OKBLUE
                        print(color + self.vocab.vocab[word] + util.c.ENDC, end=' ')
                    print()
                input('\n\nPress enter to continue ...\n')
