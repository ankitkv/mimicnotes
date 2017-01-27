from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reader


class Runner(object):
    '''Base class for all runners.'''

    def __init__(self, config, vocab, session, train_splits=['train'], val_splits=['val'],
                 test_splits=['test']):
        self.config = config
        self.vocab = vocab
        self.session = session
        self.reader = reader.NoteICD9Reader(config, vocab)
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits

    def run(self, verbose=True):
        epoch = 1
        while True:
            if verbose:
                print('\nEpoch', epoch)
            if self.config.epochs > 0 and epoch >= self.config.epochs:
                break
            loss = self.run_epoch(epoch, self.train_splits, verbose=verbose)
            if verbose:
                print('Epoch %d: Train loss: %.4f' % (epoch, loss))
            loss = self.run_epoch(epoch, self.val_splits, train=False, verbose=verbose)
            if verbose:
                print('Epoch %d: Valid loss: %.4f' % (epoch, loss))
            epoch += 1
        loss = self.run_epoch(epoch, self.test_splits, train=False, verbose=verbose)
        if verbose:
            print('Test loss: %.4f' % (epoch, loss))

    def run_epoch(self, epoch, splits, train=True, verbose=True):
        loss = 0.0
        for step, batch in enumerate(self.reader.get(splits)):
            # expect the first element of ret to be the loss
            ret = self.run_session(batch, train=train)
            loss += ret[0]
            if verbose:
                self.verbose_output(step, ret, train=train)
            if step % self.config.print_every == 0:
                self.output(step, ret, train=train)
        return loss / (step + 1)


    def verbose_output(self, step, ret, train=True):
        pass


    def output(self, step, ret, train=True):
        pass


    def run_session(self, batch, train=True):
        raise NotImplementedError
