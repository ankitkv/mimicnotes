from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import util


class TFRunner(util.Runner):
    '''Base class for all TensorFlow runners.'''

    def __init__(self, config, session, ModelClass=None, args=None, load_embeddings=True,
                 verbose=True, train_splits=['train'], val_splits=['val'], test_splits=['test'],
                 parent_runner=None):
        super(TFRunner, self).__init__(config, train_splits=train_splits, val_splits=val_splits,
                                       test_splits=test_splits, session=session,
                                       parent_runner=parent_runner)
        if ModelClass is not None:
            if args is None:
                args = [self.config, self.vocab, self.reader.label_space_size()]
            self.model = ModelClass(*args)
            self.model.initialize(self.session, self.config.load_file)
            if load_embeddings and config.emb_file:
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
        self.loss, self.probs, self.global_step = ret[:3]
        self.labels = labels
        end = time.time()
        self.wps = n_words / (end - start)
        self.accumulate()

    def save_model(self, save_file):
        self.model.save(self.session, save_file, self.config.save_overwrite)
