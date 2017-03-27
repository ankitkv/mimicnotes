from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np

import util


class Runner(object):
    '''Base class for all runners for concept detection. Other types of runners can override methods
       to change behavior.'''

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
        self.best_score = 0.0

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
            self.plot(epoch, loss, True)
            self.initialize_losses()
            global_iter = self.run_epoch(epoch, global_iter, self.val_splits, train=False,
                                         verbose=verbose)
            loss = self.losses()
            if verbose:
                print('Epoch %d: Valid losses: ' % epoch, self.loss_str(loss))
            self.plot(epoch, loss, False)
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
        self.plot(None, loss, False)
        self.finish_epoch(None)

    def run_epoch(self, epoch, global_iter, splits, train=True, verbose=True):
        step = 0
        for step, batch in enumerate(self.reader.get(splits)):
            if self.config.max_steps > 0 and step >= self.config.max_steps:
                break
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

    def sanity_check_loss(self, losses):
        '''Check if the loss we care about is within sanity bounds
           [config.sanity_min, config.sanity_max]'''
        loss, micro, macro, perclass = losses
        p, r, f, ap, auc, p8 = micro
        return ap >= self.config.sanity_min and ap <= self.config.sanity_max

    def best_val_loss(self, losses):
        '''Compare loss with the best validation loss, and return True if a new best is found'''
        loss, micro, macro, perclass = losses
        p, r, f, ap, auc, p8 = micro
        if ap >= self.best_score:
            self.best_score = ap
            return True
        else:
            return False

    def start_epoch(self, epoch):
        '''Called before the start of an epoch. epoch is None for testing (after training loop).'''
        pass

    def finish_epoch(self, epoch):
        '''Called after finishing an epoch. epoch is None for testing (after training loop).'''
        pass

    def initialize_losses(self):
        '''Initialize stuff for tracking losses'''
        self.all_losses = []
        self.all_probs = []
        self.all_labels = []

    def accumulate(self):
        self.all_losses.append(self.loss)
        self.all_probs.append(self.probs)
        self.all_labels.append(self.labels)

    def losses(self):
        '''Return the accumulated losses'''
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
        # non-avereged stats for plotting
        p, r, f = util.f1_score(probs, labels, 0.5, average=None)
        ap = util.auc_pr(probs, labels, average=None)
        try:
            auc = util.auc_roc(probs, labels, average=None)
        except ValueError:
            auc = float('nan')
        p8 = util.precision_at_k(probs, labels, 8, average=None)
        perclass = [p, r, f, ap, auc, p8]
        return loss, micro, macro, perclass

    def save_model(self, save_file):
        pass

    def loss_str(self, losses):
        loss, micro, macro, perclass = losses
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

    def verbose_output(self, step, train=True):
        pass

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

    def plot(self, epoch, losses, train, verbose=True):
        # save plot info only when testing
        if epoch is None and self.config.plot_file:
            plot_name = self.config.plot_name
            if not plot_name:
                plot_name = self.config.runner
            loss, micro, macro, perclass = losses
            with open(self.config.plot_file, 'wb') as f:
                pickle.dump((plot_name, perclass), f, -1)
            if verbose:
                print('Dumped plot info to', self.config.plot_file)

    def run_session(self, notes, lengths, labels, train=True):
        raise NotImplementedError

    def visualize(self, verbose=True):
        '''Run visualizations'''
        raise NotImplementedError
