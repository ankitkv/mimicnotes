from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, notes, lengths):
        inputs = self.embedding(notes)
        inputs = inputs.cuda()
        if self.config.rnn_type == 'gru':
            _, last_state = self.rnn(inputs, Variable(self.zero_state))
        elif self.config.rnn_type == 'lstm':
            _, last_state = self.rnn(inputs, (Variable(self.zero_h), Variable(self.zero_c)))
            last_state = torch.cat(last_state, 2)
        last_state = last_state.squeeze(0)
        logits = self.dist(last_state)
        return logits


class RecurrentNetworkTorchRunner(util.TorchRunner):
    '''Runner for the Torch recurrent network model.'''

    def __init__(self, config, verbose=True):
        super(RecurrentNetworkTorchRunner, self).__init__(config, RecurrentNetworkTorchModel)
