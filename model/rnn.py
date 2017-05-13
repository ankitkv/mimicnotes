from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from pathlib import Path
import shelve
from six.moves import xrange
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import tensorflow as tf

import model
import util

try:
    input = raw_input
except NameError:
    pass


class DynamicMemoryCell(tf.contrib.rnn.RNNCell):
    """
    Implementation of a dynamic memory cell as a gated recurrent network.
    The cell's hidden state is divided into blocks and each block's weights are tied.

    Based on https://github.com/jimfleming/recurrent-entity-networks.
    """

    def __init__(self, num_blocks, num_units_per_block, keys, initializer=None,
                 activation=tf.nn.relu):
        super(DynamicMemoryCell, self).__init__()
        self._num_blocks = num_blocks  # M
        self._num_units_per_block = num_units_per_block  # d
        self._keys = keys
        self._activation = activation  # \phi
        self._initializer = initializer

    @property
    def state_size(self):
        return self._num_blocks * self._num_units_per_block

    @property
    def output_size(self):
        return self._num_blocks * self._num_units_per_block

    def zero_state(self, batch_size, dtype):
        """
        We initialize the memory to the key values.
        """
        zero_state = tf.concat(axis=1, values=[tf.expand_dims(key, 0) for key in self._keys])
        zero_state_batch = tf.tile(zero_state, tf.stack([batch_size, 1]))
        return zero_state_batch

    def get_gate(self, state_j, key_j, inputs):
        """
        Implements the gate (scalar for each block). Equation 2:

        g_j <- \sigma(s_t^T h_j + s_t^T w_j)
        """
        a = tf.reduce_sum(inputs * state_j, axis=[1])
        b = tf.reduce_sum(inputs * tf.expand_dims(key_j, 0), axis=[1])
        return tf.sigmoid(a + b)

    def get_candidate(self, state_j, key_j, inputs, U, V, W):
        """
        Represents the new memory candidate that will be weighted by the
        gate value and combined with the existing memory. Equation 3:

        h_j^~ <- \phi(U h_j + V w_j + W s_t)
        """
        key_V = tf.matmul(tf.expand_dims(key_j, 0), V)
        state_U = tf.matmul(state_j, U)
        inputs_W = tf.matmul(inputs, W)
        return self._activation(state_U + key_V + inputs_W)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):
            # Split the hidden state into blocks (each U, V, W are shared across blocks).
            state = tf.split(axis=1, num_or_size_splits=self._num_blocks, value=state)

            U = tf.get_variable('U', [self._num_units_per_block, self._num_units_per_block])
            V = tf.get_variable('V', [self._num_units_per_block, self._num_units_per_block])
            W = tf.get_variable('W', [self._num_units_per_block, self._num_units_per_block])

            next_states = []
            for j, state_j in enumerate(state):  # Hidden State (j)
                key_j = self._keys[j]
                gate_j = self.get_gate(state_j, key_j, inputs)
                candidate_j = self.get_candidate(state_j, key_j, inputs, U, V, W)

                # Equation 4: h_j <- h_j + g_j * h_j^~
                # Perform an update of the hidden state (memory).
                state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate_j

                # Equation 5: h_j <- h_j / \norm{h_j}
                # Forget previous memories by normalization.
                state_j_next = tf.nn.l2_normalize(state_j_next, -1)

                next_states.append(state_j_next)
            state_next = tf.concat(axis=1, values=next_states)
        return state_next, state_next


class RecurrentNetworkModel(model.TFModel):
    '''A recurrent network model.'''

    def __init__(self, config, vocab, label_space_size, verbose=True):
        super(RecurrentNetworkModel, self).__init__(config, vocab, label_space_size)
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

        if config.rnn_grnn_size:
            C = config.hidden_size
            G = label_space_size
            E = config.word_emb_size
            N = np.square(C) + C + (2*C*G) + (C*E) + (G*E) + (2*G)
            hidden_size = int((np.sqrt(np.square(E+1+(G/3)) - (4*((G/3) - N))) - (E+1+(G/3))) / 2)
            if verbose:
                print('Computed RNN hidden size:', hidden_size)
        else:
            hidden_size = config.hidden_size

        if config.rnn_type == 'entnet':
            keys = [tf.get_variable('key_{}'.format(j), [config.word_emb_size])
                    for j in range(config.num_blocks)]
            cell = DynamicMemoryCell(config.num_blocks, config.word_emb_size, keys,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     activation=util.prelu)
        elif config.rnn_type == 'gru':
            cell = tf.contrib.rnn.GRUCell(hidden_size)
        elif config.rnn_type == 'lstm':
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)

        # recurrence
        initial_state = cell.zero_state(config.batch_size, tf.float32)
        outs, last_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=self.lengths,
                                             initial_state=initial_state)
        if config.rnn_type == 'lstm':
            last_state = tf.concat([s for s in last_state], 1)

        if config.rnn_type == 'entnet' and config.use_attention:
            # start with uniform attention
            attention = tf.get_variable('attention', [label_space_size, config.num_blocks],
                                        initializer=tf.zeros_initializer())
            self.attention = tf.nn.softmax(attention)

            # replicate each column of the attention matrix emb_size times (11112222...)
            attention = tf.tile(self.attention, [1, config.word_emb_size])
            attention = tf.reshape(attention, [label_space_size, config.word_emb_size,
                                               config.num_blocks])
            attention = tf.transpose(attention, [0, 2, 1])
            attention = tf.reshape(attention, [label_space_size, -1])

            # weight matrix from emb_size to label_space_size. this is the weight matrix that acts
            # on the post-attention embeddings from last_state.
            weight = tf.get_variable('weight', [label_space_size, config.word_emb_size],
                                     initializer=tf.contrib.layers.xavier_initializer())

            # tile the weight matrix num_blocks times in the second dimension and multiply the
            # attention to it. this is equivalent to doing attention + sum over all the blocks for
            # each label.
            weight = tf.tile(weight, [1, config.num_blocks])
            attended_weight = weight * attention

            # label bias
            bias = tf.get_variable("bias", [label_space_size], initializer=tf.zeros_initializer())

            logits = tf.nn.bias_add(tf.matmul(last_state, tf.transpose(attended_weight)), bias)
        else:
            logits = util.linear(last_state, self.label_space_size)
            if config.rnn_type == 'gru':
                flat_outs = tf.reshape(outs, [-1, hidden_size])
                flat_logits = util.linear(flat_outs, self.label_space_size, reuse=True)
                step_logits = tf.reshape(flat_logits, [config.batch_size, -1,
                                                       self.label_space_size])
                self.step_probs = tf.sigmoid(step_logits)

        self.probs = tf.sigmoid(logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.labels))
        self.train_op = self.minimize_loss(self.loss)


class RecurrentNetworkRunner(util.TFRunner):
    '''Runner for the recurrent network model.'''

    def __init__(self, config, session, ModelClass=RecurrentNetworkModel):
        super(RecurrentNetworkRunner, self).__init__(config, session, ModelClass=ModelClass)

    def visualize(self, verbose=True, color_changes=True, model=None, dropout=False):
        if self.config.query:
            split = self.config.query
        else:
            split = 'test'
        if model is None:
            model = self.model
        if self.config.vis_file:
            print('Preparing visualizations for dump')
            vis_info = shelve.open(self.config.vis_file, 'c', protocol=-1, writeback=True)
            count = 0
        for idx, batch in enumerate(self.reader.get([split], curriculum=False, deterministic=True)):
            if self.config.vis_file and idx % self.config.print_every == 0:
                print(idx)
            ops = [model.probs, model.step_probs]
            fdict = {model.notes: batch[0], model.lengths: batch[1], model.labels: batch[2]}
            if dropout:
                fdict[model.keep_prob] = 1.0
            probs, step_probs = self.session.run(ops, feed_dict=fdict)
            for i in xrange(probs.shape[0]):
                doc_probs = step_probs[i]  # seq_len x labels
                if self.config.vis_file:
                    doc_info = {}
                    doc_info['preds'] = []
                    doc_info['golds'] = batch[2][i]
                    for k, wordidx in enumerate(batch[0][i, :batch[1][i]]):
                        word = self.vocab.vocab[wordidx]
                        word_probs = doc_probs[k]
                        doc_info['preds'].append((word, word_probs))
                    vis_info[str(count)] = doc_info
                    count += 1
                    continue
                print()
                print('=== NEW NOTE ===')
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
        if self.config.vis_file:
            label_info = []
            for j in xrange(self.reader.label_space_size()):
                label_info.append(self.vocab.aux_names['dgn'][self.vocab.aux_vocab['dgn'][j]])
            filename = self.config.vis_file + '.labels'
            with Path(filename).open('wb') as f:
                pickle.dump(label_info, f, -1)
            print('Dumped visualization labels to', filename)
