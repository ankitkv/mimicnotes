from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import util


class SlicingGRNNModel(nn.Module):
    '''A slicing grounded RNN model.'''

    def __init__(self, config, vocab, label_space_size):
        super(SlicingGRNNModel, self).__init__()
        self.config = config
        self.label_space_size = label_space_size
        self.embedding = nn.Embedding(len(vocab.vocab), config.word_emb_size, padding_idx=0)
        hidden_size = label_space_size + config.hidden_size
        self.rnn = nn.GRU(input_size=config.word_emb_size, hidden_size=hidden_size,
                          batch_first=True)
        self.zero_state = torch.zeros([1, config.batch_size, hidden_size]).cuda()
        self.bias = nn.Parameter(torch.Tensor(label_space_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()

    def forward(self, notes, lengths):
        inputs = self.embedding(notes)
        inputs = inputs.cuda()
        _, last_state = self.rnn(inputs, Variable(self.zero_state))
        last_state = last_state.squeeze(0)
        unscaled = last_state[:, :self.label_space_size]
        scaled = (unscaled + 1) / 2
        bias = F.tanh(self.bias) * -np.log(1e-6)
        exp_nbias = torch.exp(-bias).unsqueeze(0).expand_as(scaled)
        resigmoid = scaled / (scaled + ((1.0 - scaled) * exp_nbias))
        return resigmoid


class SlicingGRNNRunner(util.TorchRunner):
    '''Runner for the slicing GRNN.'''

    def __init__(self, config, verbose=True):
        super(SlicingGRNNRunner, self).__init__(config, SlicingGRNNModel)
