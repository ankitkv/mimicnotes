from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
try:
    import cPickle as pickle
except:
    import pickle
from six.moves import xrange

import numpy as np

import util


class Runner(object):
    '''Base class for all runners for concept detection. Other types of runners can override methods
       to change behavior.'''

    def __init__(self, config, train_splits=['train'], val_splits=['val'], test_splits=['test'],
                 session=None, parent_runner=None):
        self.config = config
        self.session = session
        if parent_runner is not None:
            self.vocab = parent_runner.vocab
            self.reader = parent_runner.reader
        else:
            if config.data_storage == 'shelve':
                data = util.NoteShelveData(config)
            elif config.data_storage == 'pickle':
                data = util.NotePickleData(config)
            self.vocab = util.NoteVocab(config, data)
            self.reader = util.NoteICD9Reader(config, data, self.vocab)
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits
        self.best_score = config.best_score
        if config.save_file and Path(config.save_file).is_file():
            raise ValueError('save_file already exists.')
        if config.best_save_file and Path(config.best_save_file).is_file():
            raise ValueError('best_save_file already exists.')

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
            loss = self.losses(train=True)
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
        loss = self.losses(perclass=True)
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
        p, r, f, ap, auc = micro[:5]
        return ap >= self.config.sanity_min and ap <= self.config.sanity_max

    def best_val_loss(self, losses):
        '''Compare loss with the best validation loss, and return True if a new best is found'''
        if losses is None:
            return False
        loss, micro, macro, perclass = losses
        p, r, f, ap, auc = micro[:5]
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

    def losses(self, perclass=False, train=False, max_samples_in_chunk=(30000, 50000)):
        '''Return the accumulated losses'''
        if not self.all_losses:
            return None
        if train:
            max_samples_in_chunk = max_samples_in_chunk[0]
        else:
            max_samples_in_chunk = max_samples_in_chunk[1]
        max_batches_in_chunk = max_samples_in_chunk / self.config.batch_size
        loss = np.mean(self.all_losses)
        splits = int(0.999 + (len(self.all_probs) / max_batches_in_chunk))
        chunk_size = int(0.999 + (len(self.all_probs) / splits))
        ret_micro = []
        ret_macro = []
        ret_perclass = []
        for i in xrange(0, len(self.all_probs), chunk_size):
            all_probs = self.all_probs[i:i+chunk_size]
            all_labels = self.all_labels[i:i+chunk_size]
            probs = np.concatenate(all_probs)
            labels = np.concatenate(all_labels)
            # micro-averaged stats
            p, r, f = util.f1_score(probs, labels, 0.5)
            ap = util.auc_pr(probs, labels)
            try:
                auc = util.auc_roc(probs, labels)
            except ValueError:
                auc = float('nan')
            micro = [p, r, f, ap, auc]
            for k in self.config.pr_at_k:
                pk = util.precision_at_k(probs, labels, k)
                rk = util.recall_at_k(probs, labels, k)
                micro.extend([pk, rk])
            # macro-averaged stats
            p, r, f = util.f1_score(probs, labels, 0.5, average='macro')
            ap = util.auc_pr(probs, labels, average='macro')
            try:
                auc = util.auc_roc(probs, labels, average='macro')
            except ValueError:
                auc = float('nan')
            macro = [p, r, f, ap, auc]
            # non-avereged stats for plotting
            if perclass:
                p, r, f = util.f1_score(probs, labels, 0.5, average=None)
                ap = util.auc_pr(probs, labels, average=None)
                try:
                    auc = util.auc_roc(probs, labels, average=None)
                except ValueError:
                    auc = float('nan')
                perclass = [p, r, f, ap, auc]
            else:
                perclass = float('nan')
            ret_micro.append(micro)
            ret_macro.append(macro)
            ret_perclass.append(perclass)
            if train:
                break
        return (loss, np.mean(ret_micro, 0), np.mean(ret_macro, 0), np.mean(ret_perclass, 0))

    def save_model(self, save_file):
        pass

    def loss_str(self, losses, pastable=True):
        if losses is None:
            return "N/A"
        loss, micro, macro, perclass = losses
        loss_str = "Loss: %.4f" % loss
        p, r, f, ap, auc = micro[:5]
        prk = micro[5:]
        micro_str = "Precision (micro): %.4f, Recall (micro): %.4f, F-score (micro): %.4f, " \
                    "AUC(PR) (micro): %.4f, AUC(ROC) (micro): %.4f" % (p, r, f, ap, auc)
        pastables = ["%.4f" % n for n in (p, r, f, ap, auc)]
        p, r, f, ap, auc = macro
        macro_str = "Precision (macro): %.4f, Recall (macro): %.4f, F-score (macro): %.4f, " \
                    "AUC(PR) (macro): %.4f, AUC(ROC) (macro): %.4f" % (p, r, f, ap, auc)
        pastables.extend(["%.4f" % n for n in (p, r, f, ap, auc)])
        pr_strs = []
        for i in xrange(0, len(prk), 2):
            k = self.config.pr_at_k[i//2]
            pk, rk = prk[i], prk[i+1]
            pastables.extend(["%.4f" % n for n in (pk, rk)])
            pr_strs.append("Precision@%d: %.4f, Recall@%d: %.4f" % (k, pk, k, rk))
        pr_str = ', '.join(pr_strs)
        out_list = [loss_str, micro_str, macro_str, pr_str]
        if pastable:
            out_list.append('Pastable: ' + '\t'.join(pastables))
        return ' | '.join(out_list)

    def verbose_output(self, step, train=True):
        pass

    def output(self, step, train=True):
        p, r, f = util.f1_score(self.probs, self.labels, 0.5)
        ap = util.auc_pr(self.probs, self.labels)
        try:
            auc = util.auc_roc(self.probs, self.labels)
        except ValueError:
            auc = float('nan')
        loss_str = "GS:%d, S:%d.  Loss: %.4f, Precision: %.4f, Recall: %.4f, F-score: %.4f, " \
                   "AUC(PR): %.4f, AUC(ROC): %.4f" % (self.global_step, step, self.loss, p, r, f,
                                                      ap, auc)
        pr_strs = []
        for k in self.config.pr_at_k:
            pk = util.precision_at_k(self.probs, self.labels, k)
            rk = util.recall_at_k(self.probs, self.labels, k)
            pr_strs.append("Precision@%d: %.4f, Recall@%d: %.4f" % (k, pk, k, rk))
        pr_str = ', '.join(pr_strs)
        wps_str = "WPS: %.2f" % self.wps
        print(', '.join([loss_str, pr_str, wps_str]))

    def plot(self, epoch, losses, train, verbose=True):
        # save plot info only when testing
        if epoch is None and self.config.plot_file:
            plot_name = self.config.plot_name
            if not plot_name:
                plot_name = self.config.plot_file
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
