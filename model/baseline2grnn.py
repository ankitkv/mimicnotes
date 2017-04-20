from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time

import numpy as np
import tensorflow as tf

import model
import util


class Baseline2GRNNRunner(util.TFRunner):
    '''Runner for the baseline to grounded RNN model.'''

    def __init__(self, config, session, verbose=True):
        super(Baseline2GRNNRunner, self).__init__(config, session)
        config.sliced_grnn = True  # this is a bit hacky..
        self.model = model.GroundedRNNModel(self.config, self.vocab, config.sliced_labels,
                                            self.reader.label_space_size(),
                                            common_scope='GRNNCommon')
        self.model.initialize(self.session, self.config.load_file)
        base_config = copy.copy(config)
        base_config.load_file = config.base_file
        base_config.sanity_epoch = -1
        base_config.save_every = -1
        self.base_runner = model.BagOfWordsRunner(base_config, session)
        if config.emb_file:
            saver = tf.train.Saver([self.model.embeddings])
            saver.restore(session, config.emb_file)
            if verbose:
                print("Embeddings loaded from", config.emb_file)

    def run_session(self, notes, lengths, labels, train=True):
        n_words = lengths.sum()
        start = time.time()
        if self.config.train_base:
            base_train = train
        else:
            base_train = False
        self.base_runner.run_session(notes, lengths, labels, train=base_train)
        probs = self.base_runner.probs
        print(probs.shape)
        ops = [self.model.loss, self.model.probs, self.model.global_step]
        if train:
            ops.append(self.model.train_op)
        counts = labels.sum(0)  # TODO use probs instead
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
        feed_dict = {self.model.notes: notes, self.model.lengths: lengths,
                     self.model.slicing_indices: indices, self.model.labels: labels}
        if train:
            feed_dict[self.model.keep_prob] = 1.0 - self.config.dropout
        else:
            feed_dict[self.model.keep_prob] = 1.0
        ret = self.session.run(ops, feed_dict=feed_dict)
        self.loss, self.probs, self.global_step = ret[:3]
        self.labels = labels
        end = time.time()
        self.wps = n_words / (end - start)
        self.accumulate()


    # WRAPPERS AROUND COMMON RUNNER FUNCTIONS:

    def initialize_losses(self):
        super(Baseline2GRNNRunner, self).initialize_losses()
        self.base_runner.initialize_losses()

    def save_model(self, save_file):
        super(Baseline2GRNNRunner, self).save_model(save_file)
        self.base_runner.save_model(save_file + '.base')

    def start_epoch(self, epoch):
        super(Baseline2GRNNRunner, self).start_epoch(epoch)
        self.base_runner.start_epoch(epoch)

    def finish_epoch(self, epoch):
        super(Baseline2GRNNRunner, self).finish_epoch(epoch)
        self.base_runner.finish_epoch(epoch)

    def losses(self):
        return super(Baseline2GRNNRunner, self).losses(), self.base_runner.losses()

    def sanity_check_loss(self, losses):
        return super(Baseline2GRNNRunner, self).sanity_check_loss(losses[0])

    def best_val_loss(self, losses):
        return super(Baseline2GRNNRunner, self).best_val_loss(losses[0])

    def loss_str(self, all_losses):
        if all_losses is None:
            return "N/A"
        losses, base_losses = all_losses
        return 'MAIN: ' + super(Baseline2GRNNRunner, self).loss_str(losses) + \
               ' ||  BASE: ' + self.base_runner.loss_str(base_losses)

    def output(self, step, train=True):
        print('[Base]', end=' ')
        self.base_runner.output(step, train)
        print('[Main]', end=' ')
        super(Baseline2GRNNRunner, self).output(step, train)
