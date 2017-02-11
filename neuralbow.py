from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from config import Config
import model
import runner
import utils


class NeuralBagOfWordsModel(model.Model):
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
        logits = utils.linear(data, self.label_space_size)
        self.probs = tf.sigmoid(logits)
        self.preds = tf.to_int32(tf.greater_equal(logits, 0.0))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.labels))
        self.train_op = self.minimize_loss(self.loss)

    def summarize(self, embed, normalize=False):
        added = tf.reduce_sum(embed, 1)
        if normalize:
            added = tf.nn.l2_normalize(added, 1)
        return added


class NeuralBagOfWordsRunner(runner.Runner):
    '''Runner for the neural bag of words model.'''

    def __init__(self, config, session, model_class=NeuralBagOfWordsModel, verbose=True):
        super(NeuralBagOfWordsRunner, self).__init__(config, session)
        self.model = model_class(self.config, self.vocab, self.reader.label_space_size())
        self.model.initialize(self.session, self.config.load_file)
        if config.emb_file:
            saver = tf.train.Saver([self.model.embeddings])
            # try to restore a saved embedding model
            saver.restore(session, config.emb_file)
            if verbose:
                print("Embeddings loaded from", config.emb_file)

    def run_session(self, batch, train=True):
        notes = batch[0]
        lengths = batch[1]
        labels = batch[2]
        ops = [self.model.loss, self.model.preds, self.model.probs, self.model.global_step]
        if train:
            ops.append(self.model.train_op)
        ret = self.session.run(ops, feed_dict={self.model.notes: notes, self.model.lengths: lengths,
                                               self.model.labels: labels})
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
        raise NotImplementedError  # TODO


def main(_):
    config = Config()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        NeuralBagOfWordsRunner(config, session).run()


if __name__ == '__main__':
    tf.app.run()
