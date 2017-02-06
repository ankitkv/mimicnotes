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
        if config.data_storage == 'shelve':
            data = reader.NoteShelveData(config)
        elif config.data_storage == 'pickle':
            data = reader.NotePickleData(config)
        self.vocab = reader.NoteVocab(config, data)
        self.reader = reader.NoteICD9Reader(config, data, self.vocab)
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits

    def run(self, verbose=True):
        if self.config.visualize:
            if verbose:
                print('Running visualizations.')
            self.visualize(verbose=verbose)
        else:
            self.run_loop(verbose=verbose)

    def run_loop(self, verbose=True):
        epoch = 1
        global_iter = 0
        while True:
            if self.config.epochs > 0 and epoch > self.config.epochs:
                break
            if verbose:
                print('\nEpoch', epoch)
            global_iter, loss = self.run_epoch(epoch, global_iter, self.train_splits,
                                               verbose=verbose)
            if verbose:
                try:
                    print('Epoch %d: Train losses:' % epoch, self.loss_str(loss))
                except:  # for empty splits
                    pass
            global_iter, loss = self.run_epoch(epoch, global_iter, self.val_splits, train=False,
                                               verbose=verbose)
            if verbose:
                try:
                    print('Epoch %d: Valid losses:' % epoch, self.loss_str(loss))
                except:
                    pass
            epoch += 1
        global_iter, loss = self.run_epoch(epoch, global_iter, self.test_splits, train=False,
                                           verbose=verbose)
        if verbose:
            try:
                print('Test losses:', self.loss_str(loss))
            except:
                pass

    def run_epoch(self, epoch, global_iter, splits, train=True, verbose=True):
        loss = None
        step = 0
        for step, batch in enumerate(self.reader.get(splits)):
            if train:
                global_iter += 1
            losses, extra = self.run_session(batch, train=train)
            if loss is None:
                loss = np.array(losses)
            else:
                loss += np.array(losses)
            if verbose:
                self.verbose_output(step, losses, extra, train=train)
            if step % self.config.print_every == 0:
                self.output(step, losses, extra, train=train)
            if train and self.config.save_every > 0 and global_iter % self.config.save_every == 0:
                self.save_model()
        if loss is None:
            loss = np.array([0.0])
        return global_iter, loss / (step + 1)  # problem: gives unequal weight to smaller batches

    def save_model(self):
        pass

    def loss_str(self, loss):
        return str(loss)

    def verbose_output(self, step, losses, extra, train=True):
        pass

    def output(self, step, losses, extra, train=True):
        pass

    def run_session(self, batch, train=True):
        '''Should return (losses, extra_info)'''
        raise NotImplementedError

    def visualize(self, verbose=True):
        '''Run visualizations'''
        raise NotImplementedError
