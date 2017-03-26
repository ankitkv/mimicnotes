from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
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
        self.loss, self.probs, self.global_step = ret[:3]
        self.labels = labels
        end = time.time()
        self.wps = n_words / (end - start)
        self.accumulate()

    def initialize_losses(self):
        self.all_losses = []
        self.all_probs = []
        self.all_labels = []

    def accumulate(self):
        self.all_losses.append(self.loss)
        self.all_probs.append(self.probs)
        self.all_labels.append(self.labels)

    def losses(self):
        loss = np.mean(self.all_losses)
        probs = np.concatenate(self.all_probs)
        labels = np.concatenate(self.all_labels)
        # micro-averaged stats
        p, r, f = util.f1_score(probs, labels, 0.5)
        ap = util.auc_pr(probs, labels)
        try:
            auc = util.auc_roc(probs, labels)
        except ValueError:
            auc = float('nan')
        p8 = util.precision_at_k(probs, labels, 8)
        micro = [p, r, f, ap, auc, p8]
        # macro-averaged stats
        p, r, f = util.f1_score(probs, labels, 0.5, average='macro')
        ap = util.auc_pr(probs, labels, average='macro')
        try:
            auc = util.auc_roc(probs, labels, average='macro')
        except ValueError:
            auc = float('nan')
        p8 = util.precision_at_k(probs, labels, 8, average='macro')
        macro = [p, r, f, ap, auc, p8]
        return loss, micro, macro

    def sanity_check_loss(self, losses):
        loss, micro, macro = losses
        p, r, f, ap, auc, p8 = micro
        return ap >= self.config.sanity_min and ap <= self.config.sanity_max

    def best_val_loss(self, losses):
        '''Compare loss with the best validation loss, and return True if a new best is found'''
        loss, micro, macro = losses
        p, r, f, ap, auc, p8 = micro
        if ap >= self.best_ap:
            self.best_ap = ap
            return True
        else:
            return False

    def save_model(self, save_file):
        self.model.save(self.session, save_file, self.config.save_overwrite)

    def loss_str(self, losses):
        loss, micro, macro = losses
        loss_str = "Loss: %.4f" % loss
        p, r, f, ap, auc, p8 = micro
        micro_str = "Precision (micro): %.4f, Recall (micro): %.4f, F-score (micro): %.4f, " \
                    "AUC(PR) (micro): %.4f, AUC(ROC) (micro): %.4f, Precision@8 (micro): %.4f" % \
                    (p, r, f, ap, auc, p8)
        p, r, f, ap, auc, p8 = macro
        macro_str = "Precision (macro): %.4f, Recall (macro): %.4f, F-score (macro): %.4f, " \
                    "AUC(PR) (macro): %.4f, AUC(ROC) (macro): %.4f, Precision@8 (macro): %.4f" % \
                    (p, r, f, ap, auc, p8)
        return ' | '.join([loss_str, micro_str, macro_str])

    def output(self, step, train=True):
        p, r, f = util.f1_score(self.probs, self.labels, 0.5)
        ap = util.auc_pr(self.probs, self.labels)
        try:
            auc = util.auc_roc(self.probs, self.labels)
        except ValueError:
            auc = float('nan')
        p8 = util.precision_at_k(self.probs, self.labels, 8)
        print("GS:%d, S:%d.  Loss: %.4f, Precision: %.4f, Recall: %.4f, F-score: %.4f, "
              "AUC(PR): %.4f, AUC(ROC): %.4f, Precision@8: %.4f, WPS: %.2f" %
              (self.global_step, step, self.loss, p, r, f, ap, auc, p8, self.wps))
