# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import util


class ConvEncoderModel(nn.Module):
    """Convolutional encoder"""

    def __init__(self, config, vocab, label_space_size):
        super(ConvEncoderModel, self).__init__()
        self.config = config
        self.dropout = config.dropout
        # TODO set convolutions
        convolutions = ((config.hidden_size, 3),) * config.layers
        self.embed_tokens = Embedding(len(vocab.vocab), config.word_emb_size, 0)
        self.embed_positions = Embedding(config.max_note_len + 1, config.word_emb_size, 0)

        in_channels = convolutions[0][0]
        self.fc1 = Linear(config.word_emb_size, in_channels, dropout=self.dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        for (out_channels, kernel_size) in convolutions:
            pad = (kernel_size - 1) // 2
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                ConvTBC(in_channels, out_channels * 2, kernel_size, padding=pad,
                        dropout=self.dropout))
            in_channels = out_channels
        self.fc2 = Linear(in_channels, label_space_size)

    def forward(self, tokens, lengths):
        positions = np.arange(1, tokens.size(1) + 1, dtype=np.int)
        positions = np.tile(positions[None, ...], [tokens.size(0), 1])
        positions = torch.from_numpy(positions)
        positions[tokens.data == 0] = 0
        positions = Variable(positions, volatile=tokens.volatile)

        # embed tokens and positions
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        x = x.cuda()

        x = F.dropout(x, p=self.dropout, training=self.training)

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = x if proj is None else proj(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = F.glu(x, dim=1)
            x = (x + residual) * math.sqrt(0.5)

        # B x C x T -> B x T x C
        x = x.transpose(1, 2)
        x = x.sum(1)

        x = self.fc2(x)
        return F.sigmoid(x)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)


class ConvEncoderRunner(util.TorchRunner):
    '''Runner for the convolutional encoder model.'''

    def __init__(self, config, verbose=True):
        super(ConvEncoderRunner, self).__init__(config, ConvEncoderModel)

    def initialize_model(self, model, embeddings):
        model.cuda()
        # don't waste GPU memory on embeddings
        model.embed_tokens.cpu()
        model.embed_positions.cpu()
        if embeddings is not None:
            model.embed_tokens.weight.data.copy_(torch.from_numpy(embeddings))
            if model.embed_tokens.padding_idx is not None:
                model.embed_tokens.weight.data[model.embed_tokens.padding_idx].fill_(0)
