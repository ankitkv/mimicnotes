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


class MemoryRNNModel(model.TFModel):
    '''The memory LSTM model.'''

    def __init__(self, config, vocab, label_space_size):
        super(MemoryRNNModel, self).__init__(config, vocab, label_space_size)
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
            cell = tf.contrib.rnn.GRUCell(label_space_size +
                                          (config.word_emb_size * config.num_blocks))

        # recurrence
        out, last_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=self.lengths,
                                            swap_memory=True, dtype=tf.float32)
        self.step_probs = ((out[:, :, :label_space_size] * (1 - 2*1e-6)) + 1) / 2
        self.probs = ((last_state[:, :label_space_size] * (1 - 2*1e-6)) + 1) / 2
        loss = self.labels * -tf.log(self.probs) + (1. - self.labels) * -tf.log(1. - self.probs)
        self.loss = tf.reduce_mean(loss)
        if config.lm_weight > 0.0:
            flat_out = tf.reshape(out[:, :-1,
                                      label_space_size:label_space_size + config.word_emb_size],
                                  [-1, config.word_emb_size])
            flat_targets = tf.reshape(self.notes[:, 1:], [-1])
            flat_mask = tf.to_float(flat_targets > 0)
            lm_logits = util.linear(flat_out, len(vocab.vocab))
            lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_targets,
                                                                     logits=lm_logits) * flat_mask
            lm_loss = tf.reduce_sum(lm_loss) / tf.maximum(tf.reduce_sum(flat_mask), 1.0)
            self.loss += config.lm_weight * lm_loss

        self.train_op = self.minimize_loss(self.loss)


class MemoryRNNRunner(model.RecurrentNetworkRunner):
    '''Runner for the memory LSTM model.'''

    def __init__(self, config, session):
        super(MemoryRNNRunner, self).__init__(config, session, ModelClass=MemoryRNNModel)

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
                    print('LABEL (%s):' % verdict,
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
