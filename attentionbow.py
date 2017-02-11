from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from config import Config
import neuralbow


class AttentionBagOfWordsModel(neuralbow.NeuralBagOfWordsModel):
    '''An attention-based neural bag of words model.'''

    def __init__(self, config, vocab, label_space_size):
        super(AttentionBagOfWordsModel, self).__init__(config, vocab, label_space_size)

    def summarize(self, embed, window=4):
        '''Summarize embed using attention, where the score for each word depends on the window
           ($window to the left and to the right) around the word'''
        # TODO
        return tf.reduce_sum(embed, 1)


class AttentionBagOfWordsRunner(neuralbow.NeuralBagOfWordsRunner):
    '''Runner for the attention bag of words model.'''

    def __init__(self, config, session, verbose=True):
        super(AttentionBagOfWordsRunner, self).__init__(config, session,
                                                        model_class=AttentionBagOfWordsModel,
                                                        verbose=verbose)

    def visualize(self, verbose=True):
        raise NotImplementedError  # TODO


def main(_):
    config = Config()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        AttentionBagOfWordsRunner(config, session).run()


if __name__ == '__main__':
    tf.app.run()
