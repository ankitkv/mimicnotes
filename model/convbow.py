from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
import tensorflow as tf

from model import neuralbow
import util

try:
    input = raw_input
except NameError:
    pass


class ConvolutionalBagOfWordsModel(neuralbow.NeuralBagOfWordsModel):
    '''An attention-based neural bag of words model.'''

    def __init__(self, config, vocab, label_space_size):
        super(ConvolutionalBagOfWordsModel, self).__init__(config, vocab, label_space_size)

    def summarize(self, embed):
        '''Convolve input embeddings to get context-based embeddings'''
        self.dynamic_embs = util.conv1d(embed, self.config.word_emb_size, self.config.attn_window)
        return tf.reduce_sum(self.dynamic_embs, 1)


class ConvolutionalBagOfWordsRunner(neuralbow.NeuralBagOfWordsRunner):
    '''Runner for the attention bag of words model.'''

    def __init__(self, config, session, model_class=ConvolutionalBagOfWordsModel, verbose=True):
        super(ConvolutionalBagOfWordsRunner, self).__init__(config, session=session,
                                                            model_class=model_class,
                                                            verbose=verbose)

    def visualize_extra_ops(self):
        return []

    def visualize_extra(self, fetched, note, word):
        return ''

    def visualize(self, verbose=True):
        n_labels = 5
        with tf.variable_scope('Linear', reuse=True):
            W = tf.get_variable('Matrix').eval()  # logistic regression weights (label embeddings)
        if self.config.query:
            split = self.config.query
        else:
            split = 'test'
        for batch in self.reader.get([split], curriculum=False):
            ops = [self.model.dynamic_embs, self.model.probs]
            ops.extend(self.visualize_extra_ops())
            ret = self.session.run(ops, feed_dict={self.model.notes: batch[0],
                                                   self.model.lengths: batch[1],
                                                   self.model.labels: batch[2]})
            dynamic_embs, probs = ret[:2]
            flat_embs = dynamic_embs.reshape([-1, dynamic_embs.shape[-1]])
            flat_scores = np.dot(flat_embs, W)
            all_scores = flat_scores.reshape([dynamic_embs.shape[0], dynamic_embs.shape[1], -1])
            for i in xrange(all_scores.shape[0]):
                print()
                print('=== NEW NOTE ===')
                scores = all_scores[i]
                prob = [(j, p) for j, p in enumerate(probs[i]) if p > 0.5]
                prob.sort(key=lambda x: -x[1])
                prob = prob[:n_labels]
                for label, _ in prob:
                    if batch[2][i, label]:
                        verdict = 'correct'
                    else:
                        verdict = 'incorrect'
                    print()
                    print('LABEL (%s):' % verdict,
                          self.vocab.aux_names['dgn'][self.vocab.aux_vocab['dgn'][label]])
                    print('-----')
                    for k, word in enumerate(batch[0][i, :batch[1][i]]):
                        score = scores[k, label]
                        color = util.c.OKBLUE
                        if score > 0.7:
                            color = util.c.OKGREEN
                        elif score < -0.7:
                            color = util.c.FAIL
                        print(self.vocab.vocab[word] + self.visualize_extra(ret[2:], i, k) + color +
                              ('{%.3f}' % score) + util.c.ENDC, end=' ')
                    print()
#                print()
#                print('LABELS PER WORD')
#                print('-----')
#                for k, word in enumerate(batch[0][i, :batch[1][i]]):
#                    top_scores = [(j, s) for j, s in enumerate(scores[k]) if s > 0.1]
#                    top_scores.sort(key=lambda x: -x[1])
#                    top_scores = top_scores[:n_labels]
#                    str_labels = []
#                    for label, _ in top_scores:
#                        str_labels.append(self.vocab.aux_names
#                                                       ['dgn'][self.vocab.aux_vocab['dgn'][label]])
#                    print(self.vocab.vocab[word] + utils.c.OKBLUE +
#                                          ('{%s}' % ', '.join(str_labels)) + utils.c.ENDC, end=' ')
#                print()
                input('\n\nPress enter to continue ...\n')
