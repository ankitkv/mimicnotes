from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six.moves import xrange
import time

import numpy as np
import tensorflow as tf

import model
import util

try:
    input = raw_input
except NameError:
    pass


class DiagonalGRUCell(tf.contrib.rnn.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078) with semidiagonal weights.
       Additionally, allows to be sliced based on the labels requested."""

    def __init__(self, label_space_size, control_size, total_label_space=None,
                 activation=tf.tanh, norm=1.0, keep_prob=1.0, variables={}):
        self._label_space_size = label_space_size
        self._control_size = control_size
        self._total_label_space = total_label_space
        self._activation = activation
        self._norm = norm
        self._keep_prob = keep_prob
        self._variables = variables

    @property
    def state_size(self):
        return self._label_space_size + self._control_size

    @property
    def output_size(self):
        return self._label_space_size + self._control_size

    def diagonal_linear(self, inputs, var_scope):
        """Similar to linear, but with the weight matrix restricted to be partially diagonal."""
        diagonal = self._variables[var_scope]['Diagonal']
        right_matrix = self._variables[var_scope]['RightMatrix']
        bottom_matrix = self._variables[var_scope]['BottomMatrix']
        bias = self._variables[var_scope]['Bias']
        diag_res = inputs[:, :self._label_space_size] * tf.expand_dims(diagonal, 0)
        labels_dropped = tf.nn.dropout(inputs[:, :self._label_space_size], self._keep_prob)
        res = tf.matmul(inputs[:, self._label_space_size:], right_matrix)
        res += tf.concat([diag_res, tf.matmul(labels_dropped, bottom_matrix) * self._norm], 1)
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
        if config.sliced_grnn and not test:
            self.slicing_indices = tf.placeholder(tf.int32, [label_space_size],
                                                  name='slicing_indices')
        else:
            self.slicing_indices = None

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
                initializer = tf.contrib.layers.xavier_initializer()
                variables = collections.defaultdict(dict)
                # define the variables here so that slicing happens only once per batch
                for sc_name, bias_start in [('r_gate', 1.0), ('u_gate', 1.0), ('candidate', 0.0)]:
                    with tf.variable_scope('rnn/diagonal_gru_cell/' + sc_name):
                        if config.sliced_grnn:
                            with tf.device('/cpu:0'):
                                diagonal = tf.get_variable("Diagonal", [total_label_space],
                                                           dtype=tf.float32,
                                                           initializer=tf.zeros_initializer())
                                if self.slicing_indices is not None:
                                    diagonal = tf.gather(diagonal, self.slicing_indices)
                        else:
                            diagonal = tf.get_variable("Diagonal", [total_label_space],
                                                       dtype=tf.float32,
                                                       initializer=tf.zeros_initializer())
                        if sc_name == 'candidate' and config.positive_diag:
                            diagonal = tf.nn.elu(diagonal) + 1

                        nondiag_size = config.hidden_size + inputs.get_shape()[2].value
                        if config.sliced_grnn:
                            with tf.device('/cpu:0'):
                                right_matrix = tf.get_variable("RightMatrixT",
                                                               [total_label_space +
                                                                config.hidden_size, nondiag_size],
                                                               dtype=tf.float32,
                                                               initializer=initializer)
                                if self.slicing_indices is not None:
                                    right_matrix = tf.concat([tf.gather(
                                                                   right_matrix[:total_label_space],
                                                                   self.slicing_indices),
                                                              right_matrix[total_label_space:]], 0)
                                right_matrix = tf.transpose(right_matrix)
                        else:
                            right_matrix = tf.get_variable("RightMatrix", [nondiag_size,
                                                                           total_label_space +
                                                                           config.hidden_size],
                                                           dtype=tf.float32,
                                                           initializer=initializer)

                        # it's probably a good idea to regularize the following matrix:
                        if config.sliced_grnn:
                            with tf.device('/cpu:0'):
                                bottom_matrix = tf.get_variable("BottomMatrix",
                                                                [total_label_space,
                                                                 config.hidden_size],
                                                                dtype=tf.float32,
                                                                initializer=initializer)
                                if self.slicing_indices is not None:
                                    bottom_matrix = tf.gather(bottom_matrix, self.slicing_indices)
                        else:
                            bottom_matrix = tf.get_variable("BottomMatrix", [total_label_space,
                                                                             config.hidden_size],
                                                            dtype=tf.float32,
                                                            initializer=initializer)

                        if config.sliced_grnn:
                            with tf.device('/cpu:0'):
                                bias_term = tf.get_variable("Bias",
                                                    [total_label_space + config.hidden_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.constant_initializer(bias_start))
                                if self.slicing_indices is not None:
                                    bias_term = tf.concat([tf.gather(bias_term[:total_label_space],
                                                                     self.slicing_indices),
                                                           bias_term[total_label_space:]], 0)
                        else:
                            bias_term = tf.get_variable("Bias",
                                                    [total_label_space + config.hidden_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.constant_initializer(bias_start))

                    variables[sc_name]['Diagonal'] = diagonal
                    variables[sc_name]['RightMatrix'] = right_matrix
                    variables[sc_name]['BottomMatrix'] = bottom_matrix
                    variables[sc_name]['Bias'] = bias_term

                if config.sliced_grnn and test:
                    cell = DiagonalGRUCell(total_label_space, config.hidden_size,
                                           total_label_space=total_label_space,
                                           norm=config.sliced_labels/total_label_space,
                                           variables=variables)
                else:
                    cell = DiagonalGRUCell(label_space_size, config.hidden_size,
                                           total_label_space=total_label_space,
                                           keep_prob=self.keep_prob, variables=variables)
            else:
                cell = tf.contrib.rnn.GRUCell(label_space_size + config.hidden_size)
            # forward recurrence
            out, last_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.lengths,
                                                swap_memory=True, dtype=tf.float32)

        self.probs = (last_state[:, :label_space_size] + 1) / 2
        self.step_probs = (out[:, :, :label_space_size] + 1) / 2
        if config.biased_sigmoid:
            label_bias = tf.get_variable('label_bias', [total_label_space],
                                         initializer=tf.constant_initializer(0.0))
            if config.sliced_grnn and not test:
                label_bias = tf.gather(label_bias, self.slicing_indices)
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


class GroundedRNNRunner(util.TFRunner):
    '''Runner for the grounded RNN model.'''

    def __init__(self, config, session, verbose=True):
        super(GroundedRNNRunner, self).__init__(config, session)
        if config.sliced_grnn:
            label_space_size = config.sliced_labels
        else:
            label_space_size = self.reader.label_space_size()
        self.model = GroundedRNNModel(self.config, self.vocab, label_space_size,
                                      self.reader.label_space_size())
        if config.sliced_grnn:
            tf.get_variable_scope().reuse_variables()
            self.test_model = GroundedRNNModel(self.config, self.vocab,
                                               self.reader.label_space_size(), test=True)
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
        if train or not self.config.sliced_grnn:
            model = self.model
        else:
            model = self.test_model
        ops = [model.loss, model.probs, model.global_step]
        if train:
            ops.append(model.train_op)
        feed_dict = {model.notes: notes, model.lengths: lengths}
        if self.config.sliced_grnn and train:
            counts = labels.sum(0)
            pos_indices = np.nonzero(counts)[0]
            pos_labels = self.config.sliced_labels // 2
            if pos_indices.shape[0] > pos_labels:
                np.random.shuffle(pos_indices)
                tmp_counts = counts[pos_indices]
                indices = np.argpartition(-tmp_counts, pos_labels-1)[:pos_labels]
                pos_indices = pos_indices[indices]
            neg_labels = self.config.sliced_labels - pos_indices.shape[0]
            neg_indices = np.arange(counts.shape[0], dtype=np.int)
            neg_indices = np.setdiff1d(neg_indices, pos_indices, assume_unique=True)
            if self.config.sample_uniform:
                label_probs = None
            else:
                label_freqs = self.label_freqs[neg_indices]
                label_probs = label_freqs / label_freqs.sum()
            neg_indices = np.random.choice(neg_indices, [neg_labels], replace=False, p=label_probs)
            indices = np.concatenate([pos_indices, neg_indices])
            labels = labels[:, indices]
            feed_dict[model.slicing_indices] = indices
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
        if self.config.query:
            split = self.config.query
        else:
            split = 'test'
        if self.config.sliced_grnn:
            model = self.test_model
        else:
            model = self.model
        for batch in self.reader.get([split], force_curriculum=False):
            ops = [model.probs, model.step_probs]
            probs, step_probs = self.session.run(ops, feed_dict={model.notes: batch[0],
                                                                 model.lengths: batch[1],
                                                                 model.labels: batch[2],
                                                                 model.keep_prob: 1.0})
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
                prev_prob = None
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
                        if prev_prob is None:
                            prev_prob = prob
                        diff = (prob + prev_prob) * (prob - prev_prob)
                        prev_prob = prob
                        if color_changes:
                            if diff > 0.05:
                                color = util.c.OKGREEN
                            elif diff > 0.01:
                                color = util.c.WARNING
                            elif diff > 0.0:
                                color = util.c.ENDC
                            elif diff <= -0.05:
                                color = util.c.FAIL
                            elif diff <= -0.01:
                                color = util.c.HEADER
                            elif diff <= -0.0:
                                color = util.c.OKBLUE
                        else:
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
