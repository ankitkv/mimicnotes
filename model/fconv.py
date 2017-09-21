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
import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class ConvEncoderModel(nn.Module):
    """Convolutional encoder"""

    def __init__(self, config, vocab, label_space_size, convolutions=((512, 3),) * 20):
        # TODO set convolutions
        super(ConvEncoderModel, self).__init__()
        self.config = config
        self.dropout = config.dropout
        self.num_attention_layers = None
        self.embedding = Embedding(len(vocab.vocab), config.word_emb_size, 0)
        self.embed_positions = Embedding(config.max_note_len, config.word_emb_size, 0)

        in_channels = convolutions[0][0]
        self.fc1 = Linear(config.word_emb_size, in_channels, dropout=self.dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        for (out_channels, kernel_size) in convolutions:
            pad = (kernel_size - 1) // 2  # TODO make this zero to reduce timesteps?
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                ConvTBC(in_channels, out_channels * 2, kernel_size, padding=pad,
                        dropout=self.dropout))
            in_channels = out_channels
        self.fc2 = Linear(in_channels, config.word_emb_size)

    def forward(self, tokens, positions):
        # TODO input: tokens as usual and positions as range() but with 0's for padding tokens
        # embed tokens and positions
        x = self.embedding(tokens) + self.embed_positions(positions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = x if proj is None else proj(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = F.glu(x, dim=-1)
            x = (x + residual) * math.sqrt(0.5)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        return x


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


def grad_multiply(x, scale):  # XXX needed?
    return GradMultiply.apply(x, scale)


class GradMultiply(torch.autograd.Function):  # XXX needed?
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class ConvEncoderRunner(util.TorchRunner):
    '''Runner for the convolutional encoder model.'''

    def __init__(self, config, verbose=True):
        super(ConvEncoderRunner, self).__init__(config, ConvEncoderModel)
