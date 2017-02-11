from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf

flags = tf.flags


# command-line config
flags.DEFINE_string ("data_path",    "data",              "Data path")
flags.DEFINE_string ("save_file",    "models/recent.dat", "Save file")
flags.DEFINE_string ("load_file",    "",                  "File to load model from")
flags.DEFINE_string ("emb_file",     "",                  "File to load embeddings from")
flags.DEFINE_string ("note_type",    "Discharge_summary", "The type of notes to consider")
flags.DEFINE_string ("data_storage", "pickle",            "Format to store data (shelve or pickle)")

flags.DEFINE_integer("batch_size",     128,    "Batch size")
flags.DEFINE_integer("word_emb_size",  192,    "Word embedding size")
flags.DEFINE_bool   ("train_embs",     True,   "Train word embeddings")
flags.DEFINE_integer("max_note_len",   4500,   "Maximum note length. -1 to disable")
flags.DEFINE_integer("max_dgn_labels", 500,    "Diagnoses vocabulary for labels. -1 for default")
flags.DEFINE_string ("optimizer",      "adam", "Optimizer to use (sgd, adam, adagrad, adadelta)")
flags.DEFINE_float  ("learning_rate",  1e-3,   "Optimizer initial learning rate")
flags.DEFINE_integer("threads",        -1,     "Maximum number of threads/subprocesses. -1 to "
                                               "automatically determine")
flags.DEFINE_integer("epochs",         -1,     "Number of training+val epochs. -1 for no limit, "
                                               "0 to skip to testing.")
flags.DEFINE_float  ("keep_vocab",     0.97,   "Fraction of data for vocab to cover")
flags.DEFINE_float  ("train_split",    0.9,    "Fraction of patients for training. "
                                               "test = 1 - train - val")
flags.DEFINE_float  ("val_split",      0.033,  "Fraction of patients for validation. "
                                               "test = 1 - train - val")
flags.DEFINE_integer("print_every",    50,     "Print every these many steps")
flags.DEFINE_integer("save_every",     500,    "Save every these many steps (0 to disable)")
flags.DEFINE_bool   ("save_overwrite", True,   "Overwrite the same save file each time")
flags.DEFINE_bool   ("visualize",      False,  "Run visualizations")
flags.DEFINE_string ("query",          "",     "Query for visualization")


class Config(object):
    '''This class encapsulates all the configuration for the model.'''

    def __init__(self, from_cmd_line=True, verbose=True):
        if from_cmd_line:
            if verbose:
                print('Config:')
            flags.FLAGS._parse_flags()
            if verbose:
                maxlen = max(len(k) for k in flags.FLAGS.__dict__['__flags'])

            # copy flag values to attributes of this Config object
            for k, v in sorted(flags.FLAGS.__dict__['__flags'].items(), key=lambda x: x[0]):
                setattr(self, k, v)
                if verbose:
                    print(k.ljust(maxlen + 2), v)
            if verbose:
                print()
            if self.threads == -1:
                self.threads = multiprocessing.cpu_count() - 1
                if self.threads < 1:
                    self.threads = 1
                if verbose:
                    print('Setting threads to', self.threads)
                    print()
