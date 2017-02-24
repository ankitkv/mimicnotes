from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import util


def main(_):
    config = util.Config()
    if config.data_storage == 'shelve':
        data = util.NoteShelveData(config)
    elif config.data_storage == 'pickle':
        data = util.NotePickleData(config)
    vocab = util.NoteVocab(config, data)
    util.NoteICD9Reader(config, data, vocab)
    print('All done!')


if __name__ == '__main__':
    tf.app.run()
