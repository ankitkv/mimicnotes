from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import util


class TFRunner(util.Runner):
    '''Base class for all TensorFlow runners.'''

    def __init__(self, config, session, ModelClass=None, args=None, load_embeddings=True,
                 verbose=True, train_splits=['train'], val_splits=['val'], test_splits=['test']):
        super(TFRunner, self).__init__(config, train_splits=train_splits, val_splits=val_splits,
                                       test_splits=test_splits, session=session)
        self.best_ap = 0.0
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
        self.global_step = ret[2]
        probs = ret[1]
        p, r, f = util.f1_score(probs, labels, 0.5)
        ap = util.auc_pr(probs, labels)
        auc = util.auc_roc(probs, labels)
        p8 = util.precision_at_k(probs, labels, 8)
        end = time.time()
        wps = n_words / (end - start)
        return ret[0], p, r, f, ap, auc, p8, wps

    def sanity_check_loss(self, losses):
        loss, p, r, f, ap, auc, p8, wps = losses
        return f >= self.config.sanity_min and f <= self.config.sanity_max

    def best_val_loss(self, loss):
        '''Compare loss with the best validation loss, and return True if a new best is found'''
        if loss[4] >= self.best_ap:
            self.best_ap = loss[4]
            return True
        else:
            return False

    def save_model(self, save_file):
        self.model.save(self.session, save_file, self.config.save_overwrite)

    def loss_str(self, losses):
        loss, p, r, f, ap, auc, p8, wps = losses
        return "Loss: %.4f, Precision: %.4f, Recall: %.4f, F-score: %.4f, AUC(PR): %.4f, " \
               "AUC(ROC): %.4f, Precision@8: %.4f, WPS: %.2f" % (loss, p, r, f, ap, auc, p8, wps)

    def output(self, step, train=True):
        print("GS:%d, S:%d.  %s" % (self.global_step, step, self.loss_str(self.losses())))
