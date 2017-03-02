from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

import tensorflow as tf

import util


def main(_):
    config = util.Config()
    ModelClass = getattr(importlib.import_module("model"), config.runner)
    if config.use_tensorflow:
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
            ModelClass(config, session).run()
    else:
        ModelClass(config).run()


if __name__ == '__main__':
    tf.app.run()
