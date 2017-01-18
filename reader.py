from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import random
import shelve
try:
    import cPickle as pickle
except:
    import pickle

import nltk
import numpy as np
import tensorflow as tf

from config import Config
import utils


class Vocab(object):
    '''Stores the vocab: forward and reverse mappings'''

    def __init__(self, config):
        self.config = config
        self.vocab = ['<sos>', '<eos>', '<unk>']
        self.vocab_lookup = {w: i for i, w in enumerate(self.vocab)}
        self._init_special()

    def _init_special(self):
        self.sos_index = self.vocab_lookup.get('<sos>')
        self.eos_index = self.vocab_lookup.get('<eos>')
        self.unk_index = self.vocab_lookup.get('<unk>')

    def load_by_parsing(self, verbose=True):
        '''Read the vocab from the dataset'''
        if verbose:
            print('Loading vocabulary by parsing...')
        plist_file = Path(self.config.data_path) / 'processed/patients_list.pk'
        pshelf_file = Path(self.config.data_path) / 'processed/patients.shlf'
        with plist_file.open('rb') as f:
            patients_list = pickle.load(f)
        group_size = int(0.5 + (len(patients_list) / self.config.threads))
        lists = list(utils.grouper(group_size, patients_list))
        fds = utils.mt_map(self.config.threads, utils.partial_vocab,
                           zip(lists, [str(pshelf_file)] * len(lists)))
        fd = nltk.FreqDist()
        for d in fds:
            fd.update(d)
        if verbose:
            print('Full vocabulary size:', fd.B())
        count = 0
        for k, v in fd.most_common():
            self.vocab_lookup[k] = len(self.vocab)
            self.vocab.append(k)
            count += v
            if count / fd.B() >= self.config.keep_vocab:
                break
        if verbose:
            print('Pruned vocabulary loaded, size:', len(self.vocab))

    def load_from_pickle(self, verbose=True):
        '''Read the vocab from a pickled file'''
        pkfile = Path(self.config.data_path) / ('%s.%.2f' % (self.config.vocab_file,
                                                             self.config.keep_vocab))
        try:
            if verbose:
                print('Loading vocabulary from pickle...')
            with pkfile.open('rb') as f:
                self.vocab, self.vocab_lookup = pickle.load(f)
                self._init_special()
            if verbose:
                print('Vocabulary loaded, size:', len(self.vocab))
        except IOError:
            if verbose:
                print('Error loading from pickle, attempting parsing.')
            self.load_by_parsing(verbose=verbose)
            with pkfile.open('wb') as f:
                pickle.dump([self.vocab, self.vocab_lookup], f, -1)
                if verbose:
                    print('Saved pickle file.')

    def words2idxs(self, words):
        return [self.vocab_lookup.get(w) for w in words]

    def idxs2words(self, idxs):
        return [self.vocab[idx] for idx in idxs]


class Reader(object):
    '''The reader class that '''

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        plist_file = Path(self.config.data_path) / 'processed/patients_list.pk'
        with plist_file.open('rb') as f:
            patients_list = pickle.load(f)
        self.splits = {}
        trainidx = int(self.config.train_split * len(patients_list))
        validx = trainidx + int(self.config.val_split * len(patients_list))
        self.splits['train'] = patients_list[:trainidx]
        self.splits['val'] = patients_list[trainidx:validx]
        self.splits['test'] = patients_list[validx:]
        random.seed(0)  # deterministic random

    def read_notes(self, patients_list):
        '''Read single notes from data'''
        shelf = shelve.open(str(Path(self.config.data_path) / 'processed/patients.shlf'))
        for pid in patients_list:
            if pid is None:
                break
            try:
                ipid = int(pid)
            except ValueError:
                continue
            patient = shelf[pid]
            for adm in patient.admissions.values():
                for note in adm.nte_events:
                    note_text = []
                    for sent in utils.mimic_tokenize(note.note_text):
                        note_text.extend(sent)
                    vocab_note = [self.vocab.sos_index] + self.vocab.lookup(note_text) + \
                                 [self.vocab.eos_index]
                    yield (vocab_note, (pid, adm.admission_id))

    def buffered_read_sorted_notes(self, patients_list, batches=50):
        '''Read and return a list of notes (length multiple of batch_size) worth at most $batches
           number of batches sorted in length'''
        buffer_size = self.config.batch_size * batches
        notes = []
        for note in self.read_notes(patients_list):
            notes.append(note)
            if len(notes) == buffer_size:
                notes.sort(key=lambda x: len(x[0]))
                yield notes
                notes = []
        if notes:
            notes.sort(key=lambda x: len(x[0]))
            mod = len(notes) % self.config.batch_size
            if mod != 0:
                notes = [([], (None, None)) for _ in range(self.config.batch_size - mod)] + \
                        notes
            yield notes

    def buffered_read(self, patients_list):
        '''Read packed batches from data with each batch having notes of similar lengths'''
        for note_collection in self.buffered_read_sorted_notes(patients_list):
            batches = [b for b in utils.grouper(self.config.batch_size, note_collection)]
            random.shuffle(batches)
            for batch in batches:
                yield self.pack(batch)

    def pack(self, batch_data):
        '''Pack python-list batches into numpy batches'''
        batch = batch_data[0]
        max_size = max(len(s) for s in batch)
        ret_batch = np.zeros([self.config.batch_size, max_size], dtype=np.int32)
        lengths = np.zeros([self.config.batch_size], dtype=np.int32)
        for i, s in enumerate(batch):
            ret_batch[i, :len(s)] = s
            lengths[i] = len(s)
        return (ret_batch, lengths, batch_data[1])

    def get(self, splits):
        '''Read batches from data'''
        patients_list = sum([self.splits[s] for s in splits], [])
        for batch in self.buffered_read(patients_list):
            yield batch


def main(_):
    '''Reader tests'''
    config = Config()
    vocab = Vocab(config)
    vocab.load_from_pickle()

    reader = Reader(config, vocab)
    for batch in reader.get(['train']):
        for note in batch[0]:
            print(note)
            for e in note:
                print(vocab.vocab[e], end=' ')
            print()
            print()


if __name__ == '__main__':
    tf.app.run()
