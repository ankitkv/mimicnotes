from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from config import Config
import neuralbow
import utils


class ConvolutionalBagOfWordsModel(neuralbow.NeuralBagOfWordsModel):
    '''An attention-based neural bag of words model.'''

    def __init__(self, config, vocab, label_space_size):
        super(ConvolutionalBagOfWordsModel, self).__init__(config, vocab, label_space_size)

    def summarize(self, embed):
        '''Convolve input embeddings to get context-based embeddings'''
        embed = utils.conv1d(embed, self.config.word_emb_size, self.config.attn_window)
        return tf.reduce_sum(embed, 1)


class ConvolutionalBagOfWordsRunner(neuralbow.NeuralBagOfWordsRunner):
    '''Runner for the attention bag of words model.'''

    def __init__(self, config, session, verbose=True):
        super(ConvolutionalBagOfWordsRunner, self).__init__(config, session,
                                                        model_class=ConvolutionalBagOfWordsModel,
                                                        verbose=verbose)

    def visualize(self, verbose=True):
        raise NotImplementedError  # TODO


def main(_):
    config = Config()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        ConvolutionalBagOfWordsRunner(config, session).run()


if __name__ == '__main__':
    tf.app.run()
