from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
from six.moves import xrange
import time
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

import model
import util


class BagOfWordsModel(model.TFModel):
    '''A baseline bag of words model.'''

    def __init__(self, config, vocab, label_space_size, l1_regs=None):
        super(BagOfWordsModel, self).__init__(config, vocab, label_space_size)
        self.l1_regs = l1_regs
        if config.bow_stopwords:
            stop_words = None
        else:
            stop_words = 'english'
        if config.bow_norm:
            norm = config.bow_norm
        else:
            norm = None
        self.vectorizer = TfidfVectorizer(vocabulary=self.vocab.vocab_lookup, use_idf=False,
                                          sublinear_tf=config.bow_log_tf, stop_words=stop_words,
                                          norm=norm)
        self.data = tf.placeholder(tf.float32, [None, len(vocab.vocab)], name='data')
        self.labels = tf.placeholder(tf.float32, [None, label_space_size],
                                     name='labels')
        data_size = tf.to_float(tf.shape(self.data)[0])
        self.logits = util.linear(self.data, self.label_space_size)
        self.probs = tf.sigmoid(self.logits)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=self.labels))
        self.loss += (data_size / config.batch_size) * tf.reduce_sum(self.l1_regularization())
        self.train_op = self.minimize_loss(self.loss)

    def l1_regularization(self):
        with tf.variable_scope('Linear', reuse=True):
            W = tf.get_variable('Matrix')
        norms = tf.norm(W, 1, axis=0, keep_dims=True)
        if self.l1_regs is not None:
            return norms * self.l1_regs
        else:
            return norms * self.config.l1_reg


class BagOfWordsRunner(util.TFRunner):
    '''Runner for the bag of words model.'''

    def __init__(self, config, session, model_init=True, verbose=True):
        super(BagOfWordsRunner, self).__init__(config, session)
        l1_regs = None
        if config.bow_search:
            self.all_stats = []
            if verbose:
                print('Searching for hyperparameters')
        elif config.bow_hpfile:
            with open(config.bow_hpfile, 'rb') as f:
                l1_regs = np.expand_dims(pickle.load(f), 0)
            if verbose:
                print('Loaded custom hyperparameters')
        if model_init:  # False when __init__ is called by subclasses like neural BOW
            self.model = BagOfWordsModel(self.config, self.vocab, self.reader.label_space_size(),
                                         l1_regs)
            self.model.initialize(self.session, self.config.load_file)

    def start_epoch(self, epoch):
        if self.config.bow_search:
            self.current_stats = []

    def run_session(self, notes, lengths, labels, train=True):
        n_words = lengths.sum()
        start = time.time()
        notes = notes.tolist()
        lengths = lengths.tolist()
        X_raw = []
        for note, length in zip(notes, lengths):
            if not length:
                break
            note = note[1:length-1]
            out_note = []
            for word in note:
                out_note.append(self.vocab.vocab[word])
            X_raw.append(' '.join(out_note))
        data = self.model.vectorizer.transform(X_raw, copy=False).toarray()
        labels = labels[:len(X_raw)]
        ops = [self.model.loss, self.model.probs, self.model.global_step]
        if train:
            ops.append(self.model.train_op)
        ret = self.session.run(ops, feed_dict={self.model.data: data, self.model.labels: labels})
        self.loss, self.probs, self.global_step = ret[:3]
        self.labels = labels
        # TODO remove this and use AUC(PR) to determine best hyperparameters:
        if self.config.bow_search and not train:
            prf = {}
            for thres in np.arange(0.1, 0.75, 0.1):
                prf[int(thres * 10)] = util.f1_score(self.probs, labels, thres, average=None)[-1]
            self.current_stats.append(prf)
        end = time.time()
        self.wps = n_words / (end - start)
        self.accumulate()

    def finish_epoch(self, epoch):
        if self.config.bow_search and epoch is not None:
            self.all_stats.append(self.current_stats)
            save_file = self.config.save_file or self.config.best_save_file
            save_file += '-search.pk'
            with Path(save_file).open('wb') as f:
                pickle.dump([self.config.l1_reg, self.all_stats], f, -1)
                print('Dumped stats to', save_file)

    def visualize(self, embeddings=None, verbose=True):
        '''Visualizations for a BOW model. If embeddings is None, it is treated as an identity
           matrix. This would be a learnt matrix for neural bag of words.'''
        n_labels = 15
        n_items = 15
        with tf.variable_scope('Linear', reuse=True):
            W = tf.get_variable('Matrix').eval()  # logistic regression weights
        if embeddings is not None:
            W = np.dot(embeddings, W)
        # W /= np.linalg.norm(W, axis=0, keepdims=True)
        word_indices = [i for i in xrange(n_labels)]
        if self.config.query:
            index = self.vocab.vocab_lookup.get(self.config.query, None)
            if index:
                word_indices = [index]
        # the weights are learnt based on the addition of embeddings, so the joint distribution of
        # words in notes also plays a role, which is not visualized here (e.g. words A and B may
        # always occur together with their addition perfectly aligned with a label embedding).
        # can consider 2 and 3-tuples of words for additional visualizations.
        print()
        print('\nMOST INFLUENTIAL WORDS')
        print()
        for label in xrange(n_labels):
            print()
            print('LABEL:', self.vocab.aux_names['dgn'][self.vocab.aux_vocab['dgn'][label]])
            print('-----')
            words = [(i, w) for i, w in enumerate(W[:, label].tolist())]
            words.sort(key=lambda x: -x[1])
            for index, weight in words[:n_items]:
                print(self.vocab.vocab[index].ljust(25), weight)
            print('--')
            for index, weight in words[-n_items:]:
                print(self.vocab.vocab[index].ljust(25), weight)
        print()
        print('\nOVERALL (norms)')
        print('-----')
        W_norm = np.linalg.norm(W, axis=1)
        words = [(i, w) for i, w in enumerate(W_norm.tolist())]
        words.sort(key=lambda x: -abs(x[1]))
        for index, weight in words[:n_items]:
            print(self.vocab.vocab[index].ljust(25), weight)
        print('--')
        for index, weight in words[-n_items:]:
            print(self.vocab.vocab[index].ljust(25), weight)
        print()
        print('\nMOST INFLUENCED LABELS')
        print()
        for index in word_indices:
            print()
            print('WORD:', self.vocab.vocab[index])
            print('-----')
            labels = [(i, l) for i, l in enumerate(W[index, :].tolist())]
            labels.sort(key=lambda x: -x[1])
            for index, weight in labels[:n_items]:
                print(self.vocab.aux_names['dgn'][self.vocab.aux_vocab['dgn'][index]].ljust(25),
                      weight)
            print('--')
            for index, weight in labels[-n_items:]:
                print(self.vocab.aux_names['dgn'][self.vocab.aux_vocab['dgn'][index]].ljust(25),
                      weight)
        print()
