from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import model
import util


class NeuralBagOfWordsModel(model.TFModel):
    '''A neural bag of words model.'''

    def __init__(self, config, vocab, label_space_size):
        super(NeuralBagOfWordsModel, self).__init__(config, vocab, label_space_size)
        self.notes = tf.placeholder(tf.int32, [config.batch_size, None], name='notes')
        self.lengths = tf.placeholder(tf.int32, [config.batch_size], name='lengths')
        self.labels = tf.placeholder(tf.float32, [config.batch_size, label_space_size],
                                     name='labels')
        with tf.device('/cpu:0'):
            init_width = 0.5 / config.word_emb_size
            self.embeddings = tf.get_variable('embeddings', [len(vocab.vocab),
                                                             config.word_emb_size],
                                              initializer=tf.random_uniform_initializer(-init_width,
                                                                                        init_width),
                                              trainable=config.train_embs)
            embed = tf.nn.embedding_lookup(self.embeddings, self.notes)
        embed *= tf.to_float(tf.expand_dims(tf.greater(self.notes, 0), -1))
        data = self.summarize(embed)
        logits = util.linear(data, self.label_space_size)
        self.probs = tf.sigmoid(logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.labels))
        self.train_op = self.minimize_loss(self.loss)

    def summarize(self, embed, normalize=False):
        added = tf.reduce_sum(embed, 1)
        if normalize:
            added = tf.nn.l2_normalize(added, 1)
        return added


class NeuralBagOfWordsRunner(model.BagOfWordsRunner):
    '''Runner for the neural bag of words model.'''

    def __init__(self, config, session, model_class=NeuralBagOfWordsModel, verbose=True):
        super(NeuralBagOfWordsRunner, self).__init__(config, session=session, model_init=False)
        self.model = model_class(self.config, self.vocab, self.reader.label_space_size())
        self.model.initialize(self.session, self.config.load_file)
        if config.emb_file:
            saver = tf.train.Saver([self.model.embeddings])
            # try to restore a saved embedding model
            saver.restore(session, config.emb_file)
            if verbose:
                print("Embeddings loaded from", config.emb_file)

    def run_session(self, notes, lengths, labels, train=True):
        n_words = lengths.sum()
        start = time.time()
        ops = [self.model.loss, self.model.probs, self.model.global_step]
        if train:
            ops.append(self.model.train_op)
        ret = self.session.run(ops, feed_dict={self.model.notes: notes, self.model.lengths: lengths,
                                               self.model.labels: labels})
        probs = ret[1]
        p, r, f = util.f1_score(probs, labels, self.thresholds)
        ap = util.auc_pr(probs, labels)
        auc = util.auc_roc(probs, labels)
        p8 = util.precision_at_k(probs, labels, 8)
        end = time.time()
        wps = n_words / (end - start)
        return ([ret[0], p, r, f, ap, auc, p8, wps], [ret[2]])

    def visualize(self, verbose=True):
        super(NeuralBagOfWordsRunner, self).visualize(embeddings=self.model.embeddings.eval())
