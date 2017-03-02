from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Model(object):
    '''Base class for all models.'''

    def __init__(self, config, vocab, label_space_size):
        self.config = config
        self.vocab = vocab
        self.label_space_size = label_space_size

    def initialize(self, session, load_file, verbose=True):
        '''Load a model from a saved file if needed'''
        pass

    def save(self, session, save_file, overwrite, verbose=True):
        '''Save model to file'''
        pass
