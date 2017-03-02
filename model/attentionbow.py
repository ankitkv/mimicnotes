from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model import convbow, neuralbow
import util


class AttentionBagOfWordsModel(neuralbow.NeuralBagOfWordsModel):
    '''An attention-based neural bag of words model.'''

    def __init__(self, config, vocab, label_space_size):
        super(AttentionBagOfWordsModel, self).__init__(config, vocab, label_space_size)

    def summarize(self, embed):
        '''Summarize embed using attention, where the score for each word depends on the window
           around the word (span of width config.attn_window)'''
        if self.config.attn_on_dims:  # apply attention on each embedding dimension individually
            channels = self.config.word_emb_size
        else:
            channels = 1
        scores = util.conv1d(embed, channels, self.config.attn_window)
        if self.config.sigmoid_attn:
            self.attention = tf.sigmoid(scores)
        else:
            self.attention = tf.nn.softmax(scores, 1)
        self.dynamic_embs = embed * self.attention
        return tf.reduce_sum(self.dynamic_embs, 1)


class AttentionBagOfWordsRunner(convbow.ConvolutionalBagOfWordsRunner):
    '''Runner for the attention bag of words model.'''

    def __init__(self, config, session, verbose=True):
        super(AttentionBagOfWordsRunner, self).__init__(config, session=session,
                                                        model_class=AttentionBagOfWordsModel,
                                                        verbose=verbose)

    def visualize_extra_ops(self):
        if self.config.attn_on_dims:
            return []
        else:
            return [self.model.attention]

    def visualize_extra(self, fetched, note, word):
        if self.config.attn_on_dims:
            return ''
        else:
            attention, = fetched
            prob = attention[note, word]
            if prob > 0.01:
                color = util.c.OKGREEN
            else:
                color = util.c.OKBLUE
            return color + ('[%.3f]' % prob) + util.c.ENDC
