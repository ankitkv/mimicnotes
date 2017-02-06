from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
from six.moves import xrange

import numpy as np
import tensorflow as tf

from config import Config
import model
import runner


class Word2vecModel(model.Model):
    '''A word2vec model.'''

    def __init__(self, config, vocab, notes_count, skip_window, num_skips):
        super(Word2vecModel, self).__init__(config, vocab, 1)

        # Input data.
        self.train_inputs = tf.placeholder(tf.int64, shape=[config.batch_size])
        self.train_labels = tf.placeholder(tf.int64, shape=[config.batch_size, 1])

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            init_width = 0.5 / config.word_emb_size
            # Look up embeddings for inputs.
            embeddings = tf.get_variable('embeddings', [len(vocab.vocab), config.word_emb_size],
                                         initializer=tf.random_uniform_initializer(-init_width,
                                                                                   init_width))
            embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.get_variable('nce_weights', [len(vocab.vocab), config.word_emb_size],
                                          initializer=tf.truncated_normal_initializer(
                                                      stddev=1.0 / math.sqrt(config.word_emb_size)))
            nce_biases = tf.get_variable('nce_biases', [len(vocab.vocab)],
                                         initializer=tf.zeros_initializer())

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(self.train_labels,
                                                               1,
                                                               config.batch_size,
                                                               True,
                                                               len(vocab.vocab),
                                                               unigrams=vocab.vocab_freqs(
                                                                                       notes_count))
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                  biases=nce_biases,
                                                  labels=self.train_labels,
                                                  inputs=embed,
                                                  num_sampled=config.batch_size,
                                                  num_classes=len(vocab.vocab),
                                                  sampled_values=sampled_values))
        self.train_op = self.minimize_loss(self.loss)


class Word2vecRunner(runner.Runner):
    '''Runner for the word2vec model.'''

    def __init__(self, config, session, skip_window=4, num_skips=6):
        super(Word2vecRunner, self).__init__(config, session)
        assert num_skips <= 2 * skip_window
        self.skip_window = skip_window
        self.num_skips = num_skips
        # using a notes_count of len(patients_list) only makes sense for discharge summaries!
        self.model = Word2vecModel(config, self.vocab, len(self.reader.data.patients_list),
                                   skip_window, num_skips)
        self.model.initialize(self.session, self.config.load_file)

    def run_session(self, raw_batch, train=True):
        batch = np.ndarray(shape=(self.config.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.config.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        total_loss = 0.0
        total_steps = 0
        gs = 0

        notes = raw_batch[0].tolist()
        lengths = raw_batch[1].tolist()
        batch_index = 0

        ops = [self.model.loss, self.model.global_step]
        if train:
            ops.append(self.model.train_op)

        for note, length in zip(notes, lengths):
            data = note[:length]
            if len(data) < span:
                continue
            data_index = 0
            buffer = collections.deque(maxlen=span)
            for _ in range(span):
                buffer.append(data[data_index])
                data_index += 1
            while data_index < len(data):
                target = self.skip_window  # target label at the center of the buffer
                targets_list = [j for j in xrange(span) if j != self.skip_window]
                random.shuffle(targets_list)
                for j in range(self.num_skips):
                    target = targets_list.pop()
                    batch[batch_index] = buffer[self.skip_window]
                    labels[batch_index, 0] = buffer[target]
                    batch_index += 1
                    if batch_index == self.config.batch_size:
                        ret = self.session.run(ops, feed_dict={self.model.train_inputs: batch,
                                                               self.model.train_labels: labels})
                        loss, gs = ret[0], ret[1]
                        total_loss += loss
                        total_steps += 1
                        batch_index = 0
                buffer.append(data[data_index])
                data_index += 1
        return ([total_loss / total_steps], [gs])

    def save_model(self):
        self.model.save(self.session, self.config.save_file, self.config.save_overwrite)

    def loss_str(self, losses):
        loss, = losses
        return "Loss: %.4f" % loss

    def output(self, step, losses, extra, train=True):
        global_step = extra[0]
        print("GS:%d, S:%d.  %s" % (global_step, step, self.loss_str(losses)))


def main(_):
    config = Config()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        Word2vecRunner(config, session).run()


if __name__ == '__main__':
    tf.app.run()
