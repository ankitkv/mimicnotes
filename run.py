from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

import tensorflow as tf

import util


def main(_):
    config = util.Config()
    RunnerClass = getattr(importlib.import_module("model"), config.runner)
    if issubclass(RunnerClass, util.Runner):
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
            RunnerClass(config, session).run()
    else:
        RunnerClass(config).run()


if __name__ == '__main__':
    tf.app.run()
