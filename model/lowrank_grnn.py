from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

import model


class LowRankGRUCell(tf.contrib.rnn.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078) with low-rank weights."""

    def __init__(self, label_space_size, control_size, activation=tf.tanh, norm=1.0, variables={},
                 use_attention=False, g_to_h_block=True, detach_g_to_h=False):
        self._label_space_size = label_space_size
        self._control_size = control_size
        self._activation = activation
        self._norm = norm
        self._variables = variables
        self._g_to_h_block = g_to_h_block
        self._detach_g_to_h = detach_g_to_h
        self._use_attention = use_attention

    @property
    def state_size(self):
        return self._label_space_size + self._control_size

    @property
    def output_size(self):
        return self._label_space_size + self._control_size

    def lowrank_linear(self, inputs, var_scope):
        """Similar to linear, but with the label to label block constrained to be low rank."""
        lr_factor1 = self._variables[var_scope]['LRFactor1']
        lr_factor2 = self._variables[var_scope]['LRFactor2']
        right_matrix = self._variables[var_scope]['RightMatrix']
        if self._g_to_h_block:
            bottom_matrix = self._variables[var_scope]['BottomMatrix']
        bias = self._variables[var_scope]['Bias']
        cur_labels = inputs[:, :self._label_space_size]
        lr_res = tf.matmul(tf.matmul(cur_labels, lr_factor2), lr_factor1)
        res = tf.matmul(inputs[:, self._label_space_size:], right_matrix)
        if self._g_to_h_block:
            if self._detach_g_to_h:
                cur_labels = tf.stop_gradient(cur_labels)
            res += tf.concat([lr_res, tf.matmul(cur_labels, bottom_matrix) * self._norm], 1)
        else:
            res += tf.concat([lr_res, tf.zeros([cur_labels.get_shape()[0].value,
                                                self._control_size])], 1)
        return tf.nn.bias_add(res, bias)

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or "lowrank_gru_cell"):
            if self._use_attention:
                a_state = state  # TODO use attention using inputs and control dimensions
            else:
                a_state = state
            with tf.variable_scope("r_gate"):
                r = tf.sigmoid(self.lowrank_linear(tf.concat([a_state, inputs], 1), 'r_gate'))
            with tf.variable_scope("u_gate"):
                u = tf.sigmoid(self.lowrank_linear(tf.concat([a_state, inputs], 1), 'u_gate'))
            with tf.variable_scope("candidate"):
                c = self._activation(self.lowrank_linear(tf.concat([r * state, inputs], 1),
                                                         'candidate'))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class LowRankGRNNModel(model.TFModel):
    '''The grounded RNN model.'''

    def __init__(self, config, vocab, label_space_size, scope=None):
        super(LowRankGRNNModel, self).__init__(config, vocab, label_space_size, scope=scope)
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
                with tf.variable_scope('rnn/lowrank_gru_cell/' + sc_name):
                    lr_factor1 = tf.get_variable("LRFactor1", [config.label_emb_size,
                                                               label_space_size],
                                                 dtype=tf.float32, initializer=initializer)
                    lr_factor2 = tf.get_variable("LRFactor2", [label_space_size,
                                                               config.label_emb_size],
                                                 dtype=tf.float32, initializer=initializer)

                    nondiag_size = config.hidden_size + inputs.get_shape()[2].value
                    right_matrix = tf.get_variable("RightMatrix", [nondiag_size,
                                                                   label_space_size +
                                                                   config.hidden_size],
                                                   dtype=tf.float32,
                                                   initializer=initializer)

                    if config.g_to_h_block:
                        bottom_matrix = tf.get_variable("BottomMatrix",
                                                        [label_space_size,
                                                         config.hidden_size],
                                                        dtype=tf.float32,
                                                        initializer=initializer)

                    bias_term = tf.get_variable("Bias", [label_space_size + config.hidden_size],
                                                dtype=tf.float32,
                                                initializer=tf.constant_initializer(bias_start))

                variables[sc_name]['LRFactor1'] = lr_factor1
                variables[sc_name]['LRFactor2'] = lr_factor2
                variables[sc_name]['RightMatrix'] = right_matrix
                if config.g_to_h_block:
                    variables[sc_name]['BottomMatrix'] = bottom_matrix
                variables[sc_name]['Bias'] = bias_term

            cell = LowRankGRUCell(label_space_size, config.hidden_size,
                                  variables=variables, use_attention=config.use_attention,
                                  g_to_h_block=config.g_to_h_block,
                                  detach_g_to_h=config.detach_g_to_h)
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

        # fix potential numerical instability
        self.probs = self.probs * (1 - 2*1e-6) + 1e-6
        self.step_probs = self.step_probs * (1 - 2*1e-6) + 1e-6
        loss = self.labels * -tf.log(self.probs) + (1. - self.labels) * -tf.log(1. - self.probs)
        loss = tf.abs(self.labels - self.probs)
        self.loss = tf.reduce_mean(loss)
        if self.config.l1_reg > 0.0 or self.config.l2_reg > 0.0:
            bottom_matrices = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='.*LowRankLinear/BottomMatrix')
            accumulated = tf.concat([tf.reshape(v, [-1]) for v in bottom_matrices], 0)
            self.loss += self.l1_reg(accumulated) + self.l2_reg(accumulated)

        self.train_op = self.minimize_loss(self.loss)


class LowRankGRNNRunner(model.RecurrentNetworkRunner):
    '''Runner for the low-rank grounded RNN model.'''

    def __init__(self, config, session, verbose=True):
        super(LowRankGRNNRunner, self).__init__(config, session, ModelClass=LowRankGRNNModel)
