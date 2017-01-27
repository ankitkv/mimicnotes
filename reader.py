from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
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


class NoteVocab(object):
    '''Stores the vocab: forward and reverse mappings, for text and other auxiliary info'''

    def __init__(self, config):
        self.config = config
        self.vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab_lookup = {w: i for i, w in enumerate(self.vocab)}
        self.aux_vocab = collections.defaultdict(list)
        self.aux_vocab_lookup = collections.defaultdict(dict)
        self.aux_names = collections.defaultdict(dict)
        self.sos_index = self.vocab_lookup.get('<sos>')
        self.eos_index = self.vocab_lookup.get('<eos>')
        self.unk_index = self.vocab_lookup.get('<unk>')

    def prepare_vocab_from_fd(self, vocab_fd, vocab_aux_fd, verbose=True):
        count = 0
        for k, v in vocab_fd.most_common():
            self.vocab_lookup[k] = len(self.vocab)
            self.vocab.append(k)
            count += v
            if count / vocab_fd.N() >= self.config.keep_vocab:
                break
        for key, fd in vocab_aux_fd.items():
            for k, v in fd.most_common():
                vocab = self.aux_vocab[key]
                self.aux_vocab_lookup[key][k[0]] = len(vocab)
                vocab.append(k[0])
                self.aux_names[key][k[0]] = k[1]
        if verbose:
            print('Pruned vocabulary loaded, size:', len(self.vocab))
            for k, v in self.aux_vocab.items():
                print(k + ':', len(v))

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
                           zip(lists, [(str(pshelf_file), self.config.note_type)] * len(lists)))
        vocab_fd = nltk.FreqDist()
        vocab_aux_fd = collections.defaultdict(nltk.FreqDist)
        for fd, aux_fd in fds:
            vocab_fd.update(fd)
            for k, v in aux_fd.items():
                vocab_aux_fd[k].update(v)
        if verbose:
            print('Full vocabulary size:', vocab_fd.B())
        self.prepare_vocab_from_fd(vocab_fd, vocab_aux_fd, verbose=verbose)
        return vocab_fd, vocab_aux_fd

    def load_from_pickle(self, verbose=True):
        '''Read the vocab from pickled files, saving if necessary'''
        vocab_file = self.config.vocab_file
        if self.config.note_type:
            vocab_file += '.' + self.config.note_type
        vocab_file += '.%.2f' % self.config.keep_vocab
        pkfile = Path(self.config.data_path) / vocab_file
        try:
            if verbose:
                print('Loading vocabulary from pickle...')
            with pkfile.open('rb') as f:
                self.vocab, self.vocab_lookup, \
                            self.aux_vocab, self.aux_vocab_lookup, self.aux_names = pickle.load(f)
            if verbose:
                print('Vocabulary loaded, size:', len(self.vocab))
                for k, v in self.aux_vocab.items():
                    print(k + ':', len(v))
        except IOError:
            if verbose:
                print('Error loading from pickle, processing from freq dist for new keep_vocab.')
            vocab_fd_file = self.config.vocab_fd_file
            if self.config.note_type:
                vocab_fd_file += '.' + self.config.note_type
            fdfile = Path(self.config.data_path) / vocab_fd_file
            try:
                if verbose:
                    print('Loading vocab freq dist from pickle...')
                with fdfile.open('rb') as f:
                    vocab_fd, vocab_aux_fd = pickle.load(f)
                if verbose:
                    print('Full vocabulary loaded, size:', vocab_fd.B())
                self.prepare_vocab_from_fd(vocab_fd, verbose=verbose)
            except IOError:
                if verbose:
                    print('Error loading freq dist from pickle, attempting parsing.')
                vocab_fd, vocab_aux_fd = self.load_by_parsing(verbose=verbose)
                with fdfile.open('wb') as f:
                    pickle.dump([vocab_fd, vocab_aux_fd], f, -1)
                    if verbose:
                        print('Saved vocab freq dist pickle file.')
            with pkfile.open('wb') as f:
                pickle.dump([self.vocab, self.vocab_lookup, self.aux_vocab, self.aux_vocab_lookup,
                             self.aux_names], f, -1)
                if verbose:
                    print('Saved vocab pickle file.')

    def words2idxs(self, words):
        return [self.vocab_lookup.get(w, self.unk_index) for w in words]

    def idxs2words(self, idxs):
        return [self.vocab[idx] for idx in idxs]


class NoteReader(object):
    '''The reader that yields vectorized notes and their respective patient and admission IDs
       according to the requested split(s) (train, val, test)'''

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

    def label_info(self, admission):
        '''Can be extended to provide different kinds of labels'''
        if admission is None:
            return (None, None)
        return (admission.patient_id, admission.admission_id)

    def label_space_size(self):
        return 2

    def read_notes(self, patients_list, chunk_size=200):
        '''Read single notes from data'''
        pshelf_file = Path(self.config.data_path) / 'processed/patients.shlf'
        list_chunks = utils.grouper(chunk_size, patients_list)
        for patients_list in list_chunks:
            group_size = int(0.5 + (len(patients_list) / self.config.threads))
            lists = list(utils.grouper(group_size, patients_list))
            data = utils.mt_map(self.config.threads, utils.partial_read,
                                zip(lists, [(str(pshelf_file),
                                             self.config.note_type)] * len(lists)))
            for thread_data in data:
                for note in thread_data:
                    adm, note_text = note
                    vocab_note = [self.vocab.sos_index] + self.vocab.words2idxs(note_text) + \
                                 [self.vocab.eos_index]
                    yield (vocab_note, self.label_info(adm))

    def buffered_read_sorted_notes(self, patients_list, batches=32):
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
                notes = [([], self.label_info(None))
                         for _ in range(self.config.batch_size - mod)] + notes
            yield notes

    def buffered_read(self, patients_list):
        '''Read packed batches from data with each batch having notes of similar lengths'''
        for note_collection in self.buffered_read_sorted_notes(patients_list):
            batches = [b for b in utils.grouper(self.config.batch_size, note_collection)]
            random.shuffle(batches)
            for batch in batches:
                yield self.pack(batch)

    def label_pack(self, label_info):
        '''Pack python-list label batches into numpy batches if needed'''
        return label_info

    def pack(self, batch):
        '''Pack python-list batches into numpy batches'''
        max_size = max(len(b[0]) for b in batch)
        ret_batch = np.zeros([self.config.batch_size, max_size], dtype=np.int32)
        lengths = np.zeros([self.config.batch_size], dtype=np.int32)
        for i, b in enumerate(batch):
            ret_batch[i, :len(b[0])] = b[0]
            lengths[i] = len(b[0])
        return (ret_batch, lengths, self.label_pack([b[1] for b in batch]))

    def get(self, splits, verbose=True):
        '''Read batches from data'''
        if verbose:
            print('Getting data from', '+'.join(splits), 'split')
        patients_list = sum([self.splits[s] for s in splits], [])
        for batch in self.buffered_read(patients_list):
            yield batch


class NoteICD9Reader(NoteReader):
    '''A note reader that considers ICD9 codes as labels'''

    def __init__(self, config, vocab):
        super(NoteICD9Reader, self).__init__(config, vocab)

    def label_info(self, admission):
        max_dgn_labels = len(self.vocab.aux_vocab['dgn'])
        if self.config.max_dgn_labels > 0:
            max_dgn_labels = min(max_dgn_labels, self.config.max_dgn_labels)
        label = np.zeros([max_dgn_labels], dtype=np.int)
        if admission is None:
            return label
        vocab_lookup = self.vocab.aux_vocab_lookup['dgn']
        for diag in admission.dgn_events:
            try:
                label[vocab_lookup[diag.code]] = 1
            except IndexError:
                pass
        return label

    def label_space_size(self):
        return len(self.vocab.aux_vocab['dgn'])

    def label_pack(self, label_info):
        return np.array(label_info)


def main(_):
    '''Reader tests'''
    config = Config()
    vocab = NoteVocab(config)
    vocab.load_from_pickle()

    reader = NoteICD9Reader(config, vocab)
    words = 0
    for batch in reader.get(['train']):
        for i in range(batch[0].shape[0]):
            note = batch[0][i]
            length = batch[1][i]
            label = batch[2][i]
            words += len(note)
            print(label)
            print()
#            print(note)
#            for e in note:
#                print(vocab.vocab[e], end=' ')
#            print()
#            print()
    print(words)


if __name__ == '__main__':
    tf.app.run()
