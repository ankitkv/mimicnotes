from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

import tensorflow as tf

import util


def main(_):
    config = util.Config()
    RunnerClass = getattr(importlib.import_module("model"), config.runner)
    try:
        is_torch_class = issubclass(RunnerClass, util.TorchRunner)
    except AttributeError:
        is_torch_class = False
    if is_torch_class:
        RunnerClass(config).run()
    else:
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
            RunnerClass(config, session).run()


if __name__ == '__main__':
    tf.app.run()
