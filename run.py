import importlib

import tensorflow as tf

import util


def main(_):
    config = util.Config()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    ModelClass = getattr(importlib.import_module("model"), config.runner)
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        ModelClass(config, session).run()


if __name__ == '__main__':
    tf.app.run()
