from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn

import util


class TorchRunner(util.Runner):
    '''Runner for Torch models.'''

    def __init__(self, config, ModelClass, args=None, verbose=True, train_splits=['train'],
                 val_splits=['val'], test_splits=['test']):
        super(TorchRunner, self).__init__(config, train_splits=train_splits,
                                          val_splits=val_splits, test_splits=test_splits)
        cudnn.benchmark = True
        self.best_ap = 0.0
        if args is None:
            args = [config, self.vocab, self.reader.label_space_size()]
        self.model = ModelClass(*args)
        self.criterion = nn.BCELoss()
        self.optimizer = util.torch_optimizer(config.optimizer, config.learning_rate,
                                              self.model.parameters())
        self.global_step = 0
        embeddings = None
        if config.emb_file:
            config_proto = tf.ConfigProto()
            config_proto.gpu_options.allow_growth = True
            with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
                embeddings = tf.get_variable('embeddings', [len(self.vocab.vocab),
                                                            config.word_emb_size])
                saver = tf.train.Saver([embeddings])
                # try to restore a saved embedding model
                saver.restore(session, config.emb_file)
                if verbose:
                    print("Embeddings loaded from", config.emb_file)
                embeddings = embeddings.eval()
        self.initialize_model(self.model, embeddings)
        if config.load_file:
            if verbose:
                print('Loading model from', config.load_file, '...')
            model_state_dict, optim_state_dict, self.global_step, optim_name = \
                                                                        torch.load(config.load_file)
            self.model.load_state_dict(model_state_dict)
            if config.optimizer == optim_name:
                self.optimizer.load_state_dict(optim_state_dict)
            else:
                print('warning: saved model has a different optimizer, not loading optimizer.')
            if verbose:
                print('Loaded.')

    def initialize_model(self, model, embeddings):
        model.cuda()
        model.embedding.cpu()  # don't waste GPU memory on embeddings
        if embeddings is not None:
            model.embedding.weight.data.copy_(torch.from_numpy(embeddings))
            if model.embedding.padding_idx is not None:
                model.embedding.weight.data[model.embedding.padding_idx].fill_(0)

    def run_session(self, notes, lengths, labels, train=True):
        n_words = lengths.sum()
        start = time.time()
        notes = torch.from_numpy(notes).long()
        if train:
            self.model.zero_grad()
            notes = Variable(notes)
        else:
            notes = Variable(notes, volatile=True)
        probs = self.model(notes, lengths)
        loss = self.criterion(probs, Variable(torch.from_numpy(labels).float().cuda()))
        if train:
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
        self.probs = probs.data.cpu().numpy()
        self.labels = labels
        self.loss = loss.data.cpu().numpy()
        end = time.time()
        self.wps = n_words / (end - start)
        self.accumulate()

    def initialize_losses(self):
        self.all_losses = []
        self.all_probs = []
        self.all_labels = []

    def accumulate(self):
        self.all_losses.append(self.loss)
        self.all_probs.append(self.probs)
        self.all_labels.append(self.labels)

    def losses(self):
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
        return loss, micro, macro

    def sanity_check_loss(self, losses):
        loss, micro, macro = losses
        p, r, f, ap, auc, p8 = micro
        return ap >= self.config.sanity_min and ap <= self.config.sanity_max

    def best_val_loss(self, losses):
        '''Compare loss with the best validation loss, and return True if a new best is found'''
        loss, micro, macro = losses
        p, r, f, ap, auc, p8 = micro
        if ap >= self.best_ap:
            self.best_ap = ap
            return True
        else:
            return False

    def save_model(self, save_file, verbose=True):
        if save_file:
            if not self.config.save_overwrite:
                save_file += '.' + int(self.global_step)
            if verbose:
                print('Saving model to', save_file, '...')
            with open(save_file, 'wb') as f:
                states = [self.model.state_dict(), self.optimizer.state_dict(), self.global_step,
                          self.config.optimizer]
                torch.save(states, f)
            if verbose:
                print('Saved.')

    def loss_str(self, losses):
        loss, micro, macro = losses
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
