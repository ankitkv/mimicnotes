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
        self.preds = np.zeros([config.batch_size, self.reader.label_space_size()], dtype=np.int)
        self.preds[:, :topk] = 1

    def run_session(self, notes, lengths, labels, train=True):
        p, r, f = util.f1_score(self.preds, labels)
        return ([p, r, f], [])

    def loss_str(self, losses):
        p, r, f = losses
        return "Precision: %.4f, Recall: %.4f, F-score: %.4f" % (p, r, f)

    def output(self, step, losses, extra, train=True):
        print("S:%d.  %s" % (step, self.loss_str(losses)))
