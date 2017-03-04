from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import model
import util


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

    def __init__(self, config, vocab, label_space_size):
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

        if config.rnn_type == 'entnet':
            keys = [tf.get_variable('key_{}'.format(j), [config.word_emb_size])
                    for j in range(config.num_blocks)]
            cell = DynamicMemoryCell(config.num_blocks, config.word_emb_size, keys,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     activation=util.prelu)
        elif config.rnn_type == 'gru':
            cell = tf.contrib.rnn.GRUCell(config.num_blocks * config.word_emb_size)
        elif config.rnn_type == 'lstm':
            cell = tf.contrib.rnn.BasicLSTMCell(config.num_blocks * config.word_emb_size)

        # recurrence
        initial_state = cell.zero_state(config.batch_size, tf.float32)
        _, last_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=self.lengths,
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

        self.probs = tf.sigmoid(logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.labels))
        self.train_op = self.minimize_loss(self.loss)


class RecurrentNetworkRunner(util.Runner):
    '''Runner for the recurrent network model.'''

    def __init__(self, config, session, ModelClass=RecurrentNetworkModel, verbose=True):
        super(RecurrentNetworkRunner, self).__init__(config, session=session)
        self.best_loss = float('inf')
        self.thresholds = 0.5
        self.model = ModelClass(self.config, self.vocab, self.reader.label_space_size())
        self.model.initialize(self.session, self.config.load_file)
        if config.emb_file:
            saver = tf.train.Saver([self.model.embeddings])
            # try to restore a saved embedding model
            saver.restore(session, config.emb_file)
            if verbose:
                print("Embeddings loaded from", config.emb_file)

    def run_session(self, notes, lengths, labels, train=True):
        n_words = lengths.sum()
        start = time.time()
        ops = [self.model.loss, self.model.probs, self.model.global_step]
        if train:
            ops.append(self.model.train_op)
        ret = self.session.run(ops, feed_dict={self.model.notes: notes, self.model.lengths: lengths,
                                               self.model.labels: labels})
        probs = ret[1]
        p, r, f = util.f1_score(probs, labels, self.thresholds)
        ap = util.average_precision(probs, labels)
        p8 = util.precision_at_k(probs, labels, 8)
        end = time.time()
        wps = n_words / (end - start)
        return ([ret[0], p, r, f, ap, p8, wps], [ret[2]])

    def best_val_loss(self, loss):
        '''Compare loss with the best validation loss, and return True if a new best is found'''
        if loss[0] <= self.best_loss:
            self.best_loss = loss[0]
            return True
        else:
            return False

    def save_model(self, save_file):
        self.model.save(self.session, save_file, self.config.save_overwrite)

    def loss_str(self, losses):
        loss, p, r, f, ap, p8, wps = losses
        return "Loss: %.4f, Precision: %.4f, Recall: %.4f, F-score: %.4f, AvgPrecision: %.4f, " \
               "Precision@8: %.4f, WPS: %.2f" % (loss, p, r, f, ap, p8, wps)

    def output(self, step, losses, extra, train=True):
        global_step = extra[0]
        print("GS:%d, S:%d.  %s" % (global_step, step, self.loss_str(losses)))
