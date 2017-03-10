from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import util


class GroundedRNNModel(nn.Module):
    '''Grounded recurrent network model.'''

    def __init__(self, config, vocab, label_space_size):
        super(GroundedRNNModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(len(vocab.vocab), config.word_emb_size, padding_idx=0)
        hidden_size = config.latent_size + config.hidden_size
        self.rnn = nn.GRU(input_size=config.word_emb_size, hidden_size=hidden_size,
                          batch_first=True)
        self.zero_state = torch.zeros([1, config.batch_size, hidden_size]).cuda()
        self.dist = nn.Linear(config.latent_size + 1, label_space_size, bias=False)

    def forward(self, notes, lengths):
        inputs = self.embedding(notes)
        inputs = inputs.cuda()  # TODO use packed sequences
        _, last_state = self.rnn(inputs, Variable(self.zero_state))
        last_state = last_state.squeeze(0)
        # concatenate ones to handle the bias
        latent = torch.cat([Variable(torch.ones(self.config.batch_size, 1).cuda(),
                                     requires_grad=False),
                            last_state[:, :self.config.latent_size]], 1)
        logits = self.dist(latent)
        if self.config.grnn_sigmoid:
            return F.sigmoid(logits)
        else:
            XX = latent * latent
            WW = self.dist.weight * self.dist.weight
            Xnorms = torch.sqrt(torch.sum(XX, 1))
            Wnorms = torch.sqrt(torch.sum(WW, 1))
            norms = torch.mm(Xnorms, Wnorms.transpose(1, 0))
            out = logits / norms
            return ((out * (1 - 2*1e-6)) + 1) / 2


class GroundedRNNRunner(util.TorchRunner):
    '''Runner for the grounded recurrent network model.'''

    def __init__(self, config, verbose=True):
        super(GroundedRNNRunner, self).__init__(config, GroundedRNNModel)
