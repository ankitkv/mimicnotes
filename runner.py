from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import reader


class Runner(object):
    '''Base class for all runners.'''

    def __init__(self, config, session, train_splits=['train'], val_splits=['val'],
                 test_splits=['test']):
        self.config = config
        self.session = session
        data = reader.NotePickleData(config)
        self.vocab = reader.NoteVocab(config, data)
        self.reader = reader.NoteICD9Reader(config, data, self.vocab)
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits

    def run(self, verbose=True):
        epoch = 1
        while True:
            if self.config.epochs > 0 and epoch > self.config.epochs:
                break
            if verbose:
                print('\nEpoch', epoch)
            loss = self.run_epoch(epoch, self.train_splits, verbose=verbose)
            if verbose:
                try:
                    print('Epoch %d: Train losses:' % epoch, self.loss_str(loss))
                except:  # for empty splits
                    pass
            loss = self.run_epoch(epoch, self.val_splits, train=False, verbose=verbose)
            if verbose:
                try:
                    print('Epoch %d: Valid losses:' % epoch, self.loss_str(loss))
                except:
                    pass
            epoch += 1
        loss = self.run_epoch(epoch, self.test_splits, train=False, verbose=verbose)
        if verbose:
            try:
                print('Test losses:', self.loss_str(loss))
            except:
                pass

    def run_epoch(self, epoch, splits, train=True, verbose=True):
        loss = None
        step = 0
        for step, batch in enumerate(self.reader.get(splits)):
            losses, extra = self.run_session(batch, train=train)
            if loss is None:
                loss = np.array(losses)
            else:
                loss += np.array(losses)
            if verbose:
                self.verbose_output(step, losses, extra, train=train)
            if step % self.config.print_every == 0:
                self.output(step, losses, extra, train=train)
        if loss is None:
            loss = np.array([0.0])
        return loss / (step + 1)  # problem: gives unequal weight to smaller batches

    def loss_str(self, loss):
        return str(loss)

    def verbose_output(self, step, losses, extra, train=True):
        pass

    def output(self, step, losses, extra, train=True):
        pass

    def run_session(self, batch, train=True):
        '''Should return (losses, extra_info)'''
        raise NotImplementedError
