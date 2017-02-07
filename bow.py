from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

from config import Config
import model
import runner
import utils


class BagOfWordsModel(model.Model):
    '''A baseline bag of words model.'''

    def __init__(self, config, vocab, label_space_size):
        super(BagOfWordsModel, self).__init__(config, vocab, label_space_size)
        self.vectorizer = TfidfVectorizer(vocabulary=self.vocab.vocab_lookup, use_idf=False,
                                          sublinear_tf=True)
        self.data = tf.placeholder(tf.float32, [config.batch_size, len(vocab.vocab)], name='data')
        self.labels = tf.placeholder(tf.float32, [config.batch_size, label_space_size],
                                     name='labels')
        self.logits = utils.linear(self.data, self.label_space_size)
        self.probs = tf.sigmoid(self.logits)
        self.preds = tf.to_int32(tf.greater_equal(self.logits, 0.0))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                           labels=self.labels))
        self.train_op = self.minimize_loss(self.loss)


class BagOfWordsRunner(runner.Runner):
    '''Runner for the bag of words model.'''

    def __init__(self, config, session):
        super(BagOfWordsRunner, self).__init__(config, session)
        self.model = BagOfWordsModel(self.config, self.vocab, self.reader.label_space_size())
        self.model.initialize(self.session, self.config.load_file)

    def run_session(self, batch, train=True):
        notes = batch[0].tolist()
        lengths = batch[1].tolist()
        labels = batch[2]
        X_raw = []
        for note, length in zip(notes, lengths):
            note = note[1:length-1]
            out_note = []
            for word in note:
                out_note.append(self.vocab.vocab[word])
            X_raw.append(' '.join(out_note))
        data = self.model.vectorizer.transform(X_raw, copy=False).toarray()
        ops = [self.model.loss, self.model.preds, self.model.probs, self.model.global_step]
        if train:
            ops.append(self.model.train_op)
        ret = self.session.run(ops, feed_dict={self.model.data: data, self.model.labels: labels})
        preds, probs = ret[1], ret[2]
        p, r, f = utils.f1_score(preds, labels)
        ap = utils.average_precision(probs, labels)
        p8 = utils.precision_at_k(probs, labels, 8)
        return ([ret[0], p, r, f, ap, p8], [ret[3]])

    def save_model(self):
        self.model.save(self.session, self.config.save_file, self.config.save_overwrite)

    def loss_str(self, losses):
        loss, p, r, f, ap, p8 = losses
        return "Loss: %.4f, Precision: %.4f, Recall: %.4f, F-score: %.4f, AvgPrecision: %.4f, " \
               "Precision@8: %.4f" % (loss, p, r, f, ap, p8)

    def output(self, step, losses, extra, train=True):
        global_step = extra[0]
        print("GS:%d, S:%d.  %s" % (global_step, step, self.loss_str(losses)))

    def visualize(self, verbose=True):
        n_labels = 10
        n_words = 15
        with tf.variable_scope('Linear', reuse=True):
            W = tf.get_variable('Matrix').eval()
        print()
        print('MOST INFLUENTIAL WORDS')
        print()
        for label in xrange(n_labels):
            print()
            print('LABEL:', self.vocab.aux_names['dgn'][self.vocab.aux_vocab['dgn'][label]])
            print('-----')
            words = [(i, w) for i, w in enumerate(W[:, label].tolist())]
            words.sort(key=lambda x: -abs(x[1]))
            words = words[:n_words]
            for index, weight in words:
                print(self.vocab.vocab[index].ljust(25), weight)
            print()
        print()
        print('OVERALL (norms)')
        print('-----')
        W_norm = np.linalg.norm(W, axis=1)
        words = [(i, w) for i, w in enumerate(W_norm.tolist())]
        words.sort(key=lambda x: -abs(x[1]))
        words = words[:n_words]
        for index, weight in words:
            print(self.vocab.vocab[index].ljust(25), weight)
        print()


def main(_):
    config = Config()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        BagOfWordsRunner(config, session).run()


if __name__ == '__main__':
    tf.app.run()
