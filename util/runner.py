from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import util


class Runner(object):
    '''Base class for all runners.'''

    def __init__(self, config, train_splits=['train'], val_splits=['val'], test_splits=['test'],
                 session=None):
        self.config = config
        self.session = session
        if config.data_storage == 'shelve':
            data = util.NoteShelveData(config)
        elif config.data_storage == 'pickle':
            data = util.NotePickleData(config)
        self.vocab = util.NoteVocab(config, data)
        self.reader = util.NoteICD9Reader(config, data, self.vocab)
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
        if self.config.early_stop:
            target = self.config.min_epochs
            if verbose:
                print('Initializing early stop target to', target)
        while True:
            if self.config.epochs >= 0 and epoch > self.config.epochs:
                break
            if self.config.early_stop and epoch > target:
                if verbose:
                    print('Early stopping.\n')
                break
            if verbose:
                print('\nEpoch', epoch)
            self.start_epoch(epoch)
            self.initialize_losses()
            global_iter = self.run_epoch(epoch, global_iter, self.train_splits, verbose=verbose)
            loss = self.losses()
            if verbose:
                print('Epoch %d: Train losses: ' % epoch, self.loss_str(loss))
            self.initialize_losses()
            global_iter = self.run_epoch(epoch, global_iter, self.val_splits, train=False,
                                         verbose=verbose)
            loss = self.losses()
            if verbose:
                print('Epoch %d: Valid losses: ' % epoch, self.loss_str(loss))
            if self.best_val_loss(loss):
                if verbose:
                    print('Found new best validation loss!')
                if self.config.early_stop:
                    new_target = int(epoch * self.config.stop_increment)
                    if new_target > target:
                        target = new_target
                        if verbose:
                            print('Updating early stop target to', target)
                if self.config.best_save_file:
                    self.save_model(self.config.best_save_file)
            self.finish_epoch(epoch)
            if epoch == self.config.sanity_epoch:
                if not self.sanity_check_loss(loss):
                    if verbose:
                        print('Sanity check failed, quitting.\n')
                    break
                else:
                    if verbose:
                        print('Sanity check passed.')
            epoch += 1
        self.start_epoch(None)
        self.initialize_losses()
        global_iter = self.run_epoch(epoch, global_iter, self.test_splits, train=False,
                                     verbose=verbose)
        loss = self.losses()
        if verbose:
            print('Test losses: ', self.loss_str(loss))
        self.finish_epoch(None)

    def run_epoch(self, epoch, global_iter, splits, train=True, verbose=True):
        step = 0
        for step, batch in enumerate(self.reader.get(splits)):
            if train:
                global_iter += 1
            notes, lengths, labels = batch
            self.run_session(notes, lengths, labels, train=train)
            if verbose:
                self.verbose_output(step, train=train)
            if step % self.config.print_every == 0:
                self.output(step, train=train)
            if train and self.config.save_every > 0 and global_iter % self.config.save_every == 0:
                self.save_model(self.config.save_file)
        return global_iter

    def sanity_check_loss(self, loss):
        '''Check if the loss we care about is within sanity bounds
           [config.sanity_min, config.sanity_max]'''
        return True

    def best_val_loss(self, loss):
        '''Compare loss with the best validation loss, and return True if a new best is found.
           Take care that loss may be [0.0] when the val split was empty.'''
        return False

    def start_epoch(self, epoch):
        '''Called before the start of an epoch. epoch is None for testing (after training loop).'''
        pass

    def finish_epoch(self, epoch):
        '''Called after finishing an epoch. epoch is None for testing (after training loop).'''
        pass

    def initialize_losses(self):
        '''Initialize stuff for tracking losses'''
        pass

    def losses(self):
        '''Return the accumulated losses'''
        return None

    def save_model(self, save_file):
        pass

    def loss_str(self, loss):
        return str(loss)

    def verbose_output(self, step, train=True):
        pass

    def output(self, step, train=True):
        pass

    def run_session(self, notes, lengths, labels, train=True):
        raise NotImplementedError

    def visualize(self, verbose=True):
        '''Run visualizations'''
        raise NotImplementedError
