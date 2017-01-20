from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Model(object):
    '''Base class for all models.'''

    def __init__(config, vocab):
        self.config = config
        self.vocab = vocab
