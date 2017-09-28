from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

import numpy as np
import tensorflow as tf

import model
import util


class GRNNCell(tf.contrib.rnn.RNNCell):
    """GRNN cell."""

    def __init__(self, label_space_size, total_label_space=None,
                 activation=tf.tanh, norm=1.0, keep_prob=1.0, variables={}):
        self._label_space_size = label_space_size
        self._total_label_space = total_label_space
        self._activation = activation
        self._norm = norm
        self._keep_prob = keep_prob
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
        labels_dropped = tf.nn.dropout(inputs[:, :self._label_space_size], self._keep_prob)
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


class GroundedRNNModel(model.TFModel):
    '''The grounded RNN model.'''

    def __init__(self, config, vocab, label_space_size, total_label_space=None, test=False,
                 scope=None):
        super(GroundedRNNModel, self).__init__(config, vocab, label_space_size, scope=scope)
        if total_label_space is None:
            total_label_space = label_space_size
        self.lengths = tf.placeholder(tf.int32, [config.batch_size], name='lengths')
        self.labels = tf.placeholder(tf.float32, [config.batch_size, label_space_size],
                                     name='labels')
        self.keep_prob = tf.placeholder(tf.float32)

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
        if config.multilayer:
            with tf.variable_scope('gru_rev', initializer=tf.contrib.layers.xavier_initializer()):
                rev_cell = tf.contrib.rnn.GRUCell(config.hidden_size)
                if config.bidirectional:
                    embed_ = rev_embed
                else:
                    embed_ = embed
                rev_out, _ = tf.nn.dynamic_rnn(rev_cell, embed_, sequence_length=self.lengths,
                                               swap_memory=True, dtype=tf.float32)
                if config.bidirectional:
                    rev_out = tf.reverse_sequence(rev_out, self.lengths, seq_axis=1, batch_axis=0)
            if config.reconcat_input:
                inputs = tf.concat([inputs, rev_out], 2)
            else:
                inputs = rev_out
        else:
            assert not config.bidirectional

        with tf.variable_scope('gru', initializer=tf.contrib.layers.xavier_initializer()):
            initializer = tf.contrib.layers.xavier_initializer()
            variables = collections.defaultdict(dict)
            for sc_name, bias_start in [('r_gate', 1.0), ('u_gate', 1.0), ('candidate', 0.0)]:
                with tf.variable_scope('rnn/diagonal_gru_cell/' + sc_name):
                    diagonal = tf.get_variable("Diagonal", [total_label_space],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                    nondiag_size = inputs.get_shape()[2].value
                    right_matrix = tf.get_variable("RightMatrix", [nondiag_size,
                                                                   total_label_space],
                                                   dtype=tf.float32,
                                                   initializer=initializer)

                    bias_term = tf.get_variable("Bias", [total_label_space], dtype=tf.float32,
                                                initializer=tf.constant_initializer(bias_start))

                variables[sc_name]['Diagonal'] = diagonal
                variables[sc_name]['RightMatrix'] = right_matrix
                variables[sc_name]['Bias'] = bias_term

            cell = GRNNCell(label_space_size, total_label_space=total_label_space,
                            keep_prob=self.keep_prob, variables=variables)

            # forward recurrence
            out, last_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.lengths,
                                                swap_memory=True, dtype=tf.float32)

        self.probs = (last_state[:, :label_space_size] + 1) / 2
        self.step_probs = (out[:, :, :label_space_size] + 1) / 2
        if config.biased_sigmoid:
            label_bias = tf.get_variable('label_bias', [total_label_space],
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
        if not test:
            self.train_op = self.minimize_loss(self.loss)


class GroundedRNNRunner(model.RecurrentNetworkRunner):
    '''Runner for the grounded RNN model.'''

    def __init__(self, config, session, verbose=True):
        super(GroundedRNNRunner, self).__init__(config, session, ModelClass=None)
        label_space_size = self.reader.label_space_size()
        self.model = GroundedRNNModel(self.config, self.vocab, label_space_size,
                                      self.reader.label_space_size())
        self.model.initialize(self.session, self.config.load_file)
        if config.emb_file:
            saver = tf.train.Saver([self.model.embeddings])
            saver.restore(session, config.emb_file)
            if verbose:
                print("Embeddings loaded from", config.emb_file)
        self.label_freqs = np.array(self.vocab.aux_vocab_freqs('dgn'), dtype=np.float)

    def run_session(self, notes, lengths, labels, train=True):
        n_words = lengths.sum()
        start = time.time()
        model = self.model
        ops = [model.loss, model.probs, model.global_step]
        if train:
            ops.append(model.train_op)
        feed_dict = {model.notes: notes, model.lengths: lengths}
        feed_dict[model.labels] = labels
        if train:
            feed_dict[model.keep_prob] = 1.0 - self.config.dropout
        else:
            feed_dict[model.keep_prob] = 1.0
        ret = self.session.run(ops, feed_dict=feed_dict)
        self.loss, self.probs, self.global_step = ret[:3]
        self.labels = labels
        end = time.time()
        self.wps = n_words / (end - start)
        self.accumulate()

    def visualize(self, verbose=True, color_changes=True):
        model = self.model
        super(GroundedRNNRunner, self).visualize(verbose=verbose, color_changes=color_changes,
                                                 model=model, dropout=True)
