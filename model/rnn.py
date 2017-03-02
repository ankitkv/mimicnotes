from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model
import util


class RecurrentNetworkModel(model.TFModel):
    '''A baseline recurrent network model.'''

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

        # recurrence
        cell = tf.contrib.rnn.GRUCell(config.num_blocks * config.word_emb_size)
        initial_state = cell.zero_state(config.batch_size, tf.float32)
        _, last_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=self.lengths,
                                          initial_state=initial_state)

        logits = util.linear(last_state, self.label_space_size)
        self.probs = tf.sigmoid(logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.labels))
        self.train_op = self.minimize_loss(self.loss)


class RecurrentNetworkRunner(util.Runner):
    '''Runner for the recurrent network model.'''

    def __init__(self, config, session, verbose=True):
        super(RecurrentNetworkRunner, self).__init__(config, session=session)
        self.best_loss = float('inf')
        self.thresholds = 0.5
        self.model = RecurrentNetworkModel(self.config, self.vocab, self.reader.label_space_size())
        self.model.initialize(self.session, self.config.load_file)
        if config.emb_file:
            saver = tf.train.Saver([self.model.embeddings])
            # try to restore a saved embedding model
            saver.restore(session, config.emb_file)
            if verbose:
                print("Embeddings loaded from", config.emb_file)

    def run_session(self, batch, train=True):
        notes = batch[0]
        lengths = batch[1]
        labels = batch[2]
        ops = [self.model.loss, self.model.probs, self.model.global_step]
        if train:
            ops.append(self.model.train_op)
        ret = self.session.run(ops, feed_dict={self.model.notes: notes, self.model.lengths: lengths,
                                               self.model.labels: labels})
        probs = ret[1]
        p, r, f = util.f1_score(probs, labels, self.thresholds)
        ap = util.average_precision(probs, labels)
        p8 = util.precision_at_k(probs, labels, 8)
        return ([ret[0], p, r, f, ap, p8], [ret[2]])

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
        loss, p, r, f, ap, p8 = losses
        return "Loss: %.4f, Precision: %.4f, Recall: %.4f, F-score: %.4f, AvgPrecision: %.4f, " \
               "Precision@8: %.4f" % (loss, p, r, f, ap, p8)

    def output(self, step, losses, extra, train=True):
        global_step = extra[0]
        print("GS:%d, S:%d.  %s" % (global_step, step, self.loss_str(losses)))
