from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import model
import utils


class BagOfWordsModel(model.Model):
    '''A baseline bag of words model.'''

    def __init__(self, config, vocab, label_space_size):
        super(BagOfWordsModel, self).__init__(config, vocab)
        self.data = tf.placeholder(tf.float32, [config.batch_size, len(vocab.vocab)], name='data')
        self.labels = tf.placeholder(tf.float32, [config.batch_size, label_space_size],
                                     name='labels')
        preds = self.predict(self.data)
        self.loss = self.compute_loss(preds, self.labels)
        self.train_op = self.train(self.loss)

    def predict(self, inputs):
        pass

    def loss(self, pred, real):
        pass

    def train(self, loss):
        pass


class BagOfWordsRunner(runner.Runner):
    '''Runner for the bag of words model.'''

    def __init__(self):
        super(BagOfWordsRunner, self).__init__()
        self.model = BagOfWordsModel(self.config, self.vocab, self.reader.label_space_size())

    def run_session(self, batch, train=True):
        notes = batch[0]
        lengths = batch[1]
        labels = batch[2]
        batch_size = notes.shape[0]
        bow = np.zeros([batch_size, len(self.vocab.vocab)], dtype=np.float)
        for i in range(batch_size):
            for j in range(lengths[i]):
                bow[i, notes[j]] += 1.0
        f_dict = {self.model.data: bow, self.model.labels: labels}
        session.run([self.model.train_op], f_dict)


def main(_):
    BagOfWordsRunner().run()


if __name__ == '__main__':
    tf.app.run()
