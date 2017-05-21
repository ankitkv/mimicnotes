from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

import util


class RandomRunner(util.Runner):
    '''Runner for the recurrent network model.'''

    def __init__(self, config):
        super(RandomRunner, self).__init__(config)
        self.global_step = 0

    def run_session(self, notes, lengths, labels, train=True):
        n_words = lengths.sum()
        start = time.time()
        if train:
            self.global_step += 1
        self.probs = np.random.rand(*labels.shape)
        self.labels = labels
        self.loss = 0
        end = time.time()
        self.wps = n_words / (end - start)
        self.accumulate()
