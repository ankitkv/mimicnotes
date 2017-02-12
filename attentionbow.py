from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import convbow
from config import Config
import neuralbow
import utils


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
        scores = utils.conv1d(embed, channels, self.config.attn_window)
        probs = tf.nn.softmax(scores, 1)
        self.dynamic_embs = embed * probs
        return tf.reduce_sum(self.dynamic_embs, 1)


class AttentionBagOfWordsRunner(convbow.ConvolutionalBagOfWordsRunner):
    '''Runner for the attention bag of words model.'''

    def __init__(self, config, session, verbose=True):
        super(AttentionBagOfWordsRunner, self).__init__(config, session,
                                                        model_class=AttentionBagOfWordsModel,
                                                        verbose=verbose)


def main(_):
    config = Config()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        AttentionBagOfWordsRunner(config, session).run()


if __name__ == '__main__':
    tf.app.run()
