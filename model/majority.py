from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import util


class MajorityRunner(util.Runner):
    '''Runner for the majority dummy model.'''

    def __init__(self, config, topk=8):
        super(MajorityRunner, self).__init__(config, train_splits=['train', 'val', 'test'],
                                             val_splits=[], test_splits=[])
        self.probs = np.zeros([config.batch_size, self.reader.label_space_size()])
        self.probs[:, :topk] = 1.0

    def run_session(self, notes, lengths, labels, train=True):
        self.labels = labels
        self.accumulate()

    def initialize_losses(self):
        self.all_probs = []
        self.all_labels = []

    def accumulate(self):
        self.all_probs.append(self.probs)
        self.all_labels.append(self.labels)

    def losses(self, perclass=False, train=False):
        if not self.all_probs:
            return None
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
        r8 = util.recall_at_k(probs, labels, 8)
        micro = [p, r, f, ap, auc, p8, r8]
        # macro-averaged stats
        p, r, f = util.f1_score(probs, labels, 0.5, average='macro')
        ap = util.auc_pr(probs, labels, average='macro')
        try:
            auc = util.auc_roc(probs, labels, average='macro')
        except ValueError:
            auc = float('nan')
        macro = [p, r, f, ap, auc]
        return micro, macro

    def sanity_check_loss(self, loss):
        return True

    def best_val_loss(self, loss):
        return False

    def loss_str(self, losses):
        if losses is None:
            return "N/A"
        micro, macro = losses
        p, r, f, ap, auc, p8, r8 = micro
        micro_str = "Precision (micro): %.4f, Recall (micro): %.4f, F-score (micro): %.4f, " \
                    "AUC(PR) (micro): %.4f, AUC(ROC) (micro): %.4f, Precision@8 (micro): %.4f, " \
                    "Recall@8 (micro): %.4f" % (p, r, f, ap, auc, p8, r8)
        p, r, f, ap, auc, p8, r8 = macro
        macro_str = "Precision (macro): %.4f, Recall (macro): %.4f, F-score (macro): %.4f, " \
                    "AUC(PR) (macro): %.4f, AUC(ROC) (macro): %.4f" % (p, r, f, ap, auc)
        return ' | '.join([micro_str, macro_str])

    def output(self, step, train=True):
        p, r, f = util.f1_score(self.probs, self.labels, 0.5)
        ap = util.auc_pr(self.probs, self.labels)
        try:
            auc = util.auc_roc(self.probs, self.labels)
        except ValueError:
            auc = float('nan')
        p8 = util.precision_at_k(self.probs, self.labels, 8)
        r8 = util.recall_at_k(self.probs, self.labels, 8)
        print("S:%d.  Precision: %.4f, Recall: %.4f, F-score: %.4f, AUC(PR): %.4f, AUC(ROC): %.4f, "
              "Precision@8: %.4f, Recall@8: %.4f" % (step, p, r, f, ap, auc, p8, r8))
