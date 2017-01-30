from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

from config import Config
import model
import runner


class BagOfWordsModel(model.Model):
    '''A baseline bag of words model.'''

    def __init__(self, config, vocab, label_space_size):
        super(BagOfWordsModel, self).__init__(config, vocab, label_space_size)
        self.vectorizer = TfidfVectorizer(vocabulary=self.vocab.vocab_lookup, use_idf=False,
                                          sublinear_tf=True)
        self.data = tf.placeholder(tf.float32, [config.batch_size, len(vocab.vocab)], name='data')
        self.labels = tf.placeholder(tf.float32, [config.batch_size, label_space_size],
                                     name='labels')
        self.logits = self.predict(self.data)
        self.loss = tf.reduce_mean(self.compute_loss(self.logits, self.labels))
        self.train_op = self.minimize_loss(self.loss)

    def predict(self, inputs):
        W = tf.get_variable("W", [self.label_space_size, len(self.vocab.vocab)],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [self.label_space_size], initializer=tf.zeros_initializer())
        return tf.nn.bias_add(tf.matmul(inputs, tf.transpose(W)), b)

    def compute_loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels),
                              1)


class BagOfWordsRunner(runner.Runner):
    '''Runner for the bag of words model.'''

    def __init__(self, config, session):
        super(BagOfWordsRunner, self).__init__(config, session)
        self.model = BagOfWordsModel(self.config, self.vocab, self.reader.label_space_size())
        self.session.run(tf.global_variables_initializer())

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
        ops = [self.model.loss, self.model.global_step]
        if train:
            ops.append(self.model.train_op)
        ret = self.session.run(ops, feed_dict={self.model.data: data, self.model.labels: labels})
        if train:
            return ret[:-1]
        else:
            return ret

    def output(self, step, ret, train=True):
        loss, global_step = ret
        print("GS:%d, S:%d.  Loss: %.4f" % (global_step, step, loss))


def main(_):
    config = Config()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        BagOfWordsRunner(config, session).run()


if __name__ == '__main__':
    tf.app.run()
