from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import Config
from reader import Reader, Vocab


class Runner(object):
    '''Base class for all runners.'''

    def __init__(self, train_splits=['train'], val_splits=['val'], test_splits=['test']):
        self.config = Config()
        self.vocab = Vocab(config)
        self.vocab.load_from_pickle()
        self.reader = Reader(config, vocab)
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits

    def run(self):
        epoch = 1  # TODO save in the model to resume properly
        while True:
            if self.config.epochs > 0 and epoch >= self.config.epochs:
                break
            self.run_epoch(epoch, self.train_splits)
            self.run_epoch(epoch, self.val_splits, train=False)
        self.run_epoch(epoch, self.test_splits, train=False)

    def run_epoch(self, epoch, splits, train=True):
        for step, batch in enumerate(self.reader.get(splits)):
            self.run_session(batch, train=train)

    def run_session(self, batch, train=True):
        raise NotImplementedError
