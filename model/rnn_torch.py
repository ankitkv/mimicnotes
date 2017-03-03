from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model
import util


class RecurrentNetworkTorchModel(nn.Module):
    '''A baseline recurrent network model in Torch.'''

    def __init__(self, config, vocab, label_space_size):
        super(RecurrentNetworkTorchModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(len(vocab.vocab), config.word_emb_size, padding_idx=0)
        output_size = config.word_emb_size * config.num_blocks
        if config.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=config.word_emb_size,
                              hidden_size=config.word_emb_size*config.num_blocks, batch_first=True)
            self.zero_state = torch.zeros([1, config.batch_size,
                                           config.word_emb_size * config.num_blocks]).cuda()
        elif config.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=config.word_emb_size,
                               hidden_size=config.word_emb_size*config.num_blocks, batch_first=True)
            self.zero_h = torch.zeros([1, config.batch_size,
                                       config.word_emb_size * config.num_blocks]).cuda()
            self.zero_c = torch.zeros([1, config.batch_size,
                                       config.word_emb_size * config.num_blocks]).cuda()
            output_size *= 2
        else:
            raise NotImplementedError
        self.dist = nn.Linear(output_size, label_space_size)

    def forward(self, notes):
        inputs = self.embedding(notes)
        inputs = inputs.cuda()
        if self.config.rnn_type == 'gru':
            _, last_state = self.rnn(inputs, Variable(self.zero_state))
        elif self.config.rnn_type == 'lstm':
            _, last_state = self.rnn(inputs, (Variable(self.zero_h), Variable(self.zero_c)))
            last_state = torch.cat(last_state, 2)
        last_state = last_state.squeeze(0)
        logits = self.dist(last_state)
        return F.sigmoid(logits)


class RecurrentNetworkTorchRunner(util.Runner):
    '''Runner for the Torch recurrent network model.'''

    def __init__(self, config, session, verbose=True):
        super(RecurrentNetworkTorchRunner, self).__init__(config)
        cudnn.benchmark = True
        self.best_loss = float('inf')
        self.thresholds = 0.5
        self.model = RecurrentNetworkTorchModel(config, self.vocab, self.reader.label_space_size())
        self.model.apply(util.torch_initialize)
        self.model.cuda()
        self.model.embedding.cpu()  # don't waste GPU memory on embeddings
        self.criterion = nn.BCELoss()
        self.optimizer = util.torch_optimizer(config.optimizer, config.learning_rate,
                                              self.model.parameters())
        self.global_step = 0

    def run_session(self, batch, train=True):
        notes = batch[0]
        lengths = batch[1]
        labels = batch[2]
        n_words = lengths.sum()
        notes = torch.from_numpy(notes).long()
        start = time.time()
        if train:
            self.model.zero_grad()
            notes = Variable(notes)
        else:
            notes = Variable(notes, volatile=True)
        probs = self.model(notes)
        loss = self.criterion(probs, Variable(torch.from_numpy(labels).float().cuda()))
        if train:
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
        end = time.time()
        wps = n_words / (end - start)
        probs = probs.data.cpu().numpy()
        loss = loss.data.cpu().numpy()
        p, r, f = util.f1_score(probs, labels, self.thresholds)
        ap = util.average_precision(probs, labels)
        p8 = util.precision_at_k(probs, labels, 8)
        return ([loss, p, r, f, ap, p8, wps], [])

    def best_val_loss(self, loss):
        '''Compare loss with the best validation loss, and return True if a new best is found'''
        if loss[0] <= self.best_loss:
            self.best_loss = loss[0]
            return True
        else:
            return False

    def save_model(self, save_file):
        pass  # TODO

    def loss_str(self, losses):
        loss, p, r, f, ap, p8, wps = losses
        return "Loss: %.4f, Precision: %.4f, Recall: %.4f, F-score: %.4f, AvgPrecision: %.4f, " \
               "Precision@8: %.4f, WPS: %.2f" % (loss, p, r, f, ap, p8, wps)

    def output(self, step, losses, extra, train=True):
        print("GS:%d, S:%d.  %s" % (self.global_step, step, self.loss_str(losses)))
