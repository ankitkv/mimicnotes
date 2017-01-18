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


class NoteVocab(object):
    '''Stores the vocab: forward and reverse mappings, for text and other auxiliary info'''

    def __init__(self, config):  # TODO aux vocab
        self.config = config
        self.vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab_lookup = {w: i for i, w in enumerate(self.vocab)}
        self.sos_index = self.vocab_lookup.get('<sos>')
        self.eos_index = self.vocab_lookup.get('<eos>')
        self.unk_index = self.vocab_lookup.get('<unk>')

    def prepare_vocab_from_fd(self, clear_fd=True, verbose=True):
        count = 0
        for k, v in self.vocab_fd.most_common():
            self.vocab_lookup[k] = len(self.vocab)
            self.vocab.append(k)
            count += v
            if count / self.vocab_fd.N() >= self.config.keep_vocab:
                break
        if verbose:
            print('Pruned vocabulary loaded, size:', len(self.vocab))
        if clear_fd:
            self.vocab_fd = None

    def load_by_parsing(self, prepare_vocab=True, verbose=True):
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
        self.vocab_fd = nltk.FreqDist()
        for fd in fds:
            self.vocab_fd.update(fd)
        if verbose:
            print('Full vocabulary size:', self.vocab_fd.B())
        if prepare_vocab:
            self.prepare_vocab_from_fd(verbose=verbose)

    def load_from_pickle(self, verbose=True):
        '''Read the vocab from pickled files, saving if necessary'''
        pkfile = Path(self.config.data_path) / ('%s.%.2f' % (self.config.vocab_file,
                                                             self.config.keep_vocab))
        try:
            if verbose:
                print('Loading vocabulary from pickle...')
            with pkfile.open('rb') as f:
                self.vocab, self.vocab_lookup = pickle.load(f)
            if verbose:
                print('Vocabulary loaded, size:', len(self.vocab))
        except IOError:
            if verbose:
                print('Error loading from pickle, processing from freq dist for new keep_vocab.')
            fdfile = Path(self.config.data_path) / self.config.vocab_fd_file
            try:
                if verbose:
                    print('Loading vocab freq dist from pickle...')
                with fdfile.open('rb') as f:
                    self.vocab_fd = pickle.load(f)
                if verbose:
                    print('Full vocabulary loaded, size:', self.vocab_fd.B())
            except IOError:
                if verbose:
                    print('Error loading freq dist from pickle, attempting parsing.')
                self.load_by_parsing(prepare_vocab=False, verbose=verbose)
                with fdfile.open('wb') as f:
                    pickle.dump(self.vocab_fd, f, -1)
                    if verbose:
                        print('Saved vocab freq dist pickle file.')
            self.prepare_vocab_from_fd(verbose=verbose)
            with pkfile.open('wb') as f:
                pickle.dump([self.vocab, self.vocab_lookup], f, -1)
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

    def label_info(self, patient, admission):
        '''Can be extended to provide different kinds of labels'''
        if patient is None:
            return (None, None)
        return (admission.patient_id, admission.admission_id)

    def read_notes(self, patients_list):
        '''Read single notes from data'''
        shelf = shelve.open(str(Path(self.config.data_path) / 'processed/patients.shlf'))
        for pid in patients_list:
            if pid is None:
                break
            try:
                int(pid)
            except ValueError:
                continue
            patient = shelf[pid]
            for adm in patient.admissions.values():
                for note in adm.nte_events:
                    note_text = []
                    for sent in utils.mimic_tokenize(note.note_text):
                        note_text.extend(sent)
                    vocab_note = [self.vocab.sos_index] + self.vocab.words2idxs(note_text) + \
                                 [self.vocab.eos_index]
                    yield (vocab_note, self.label_info(patient, adm))

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
                notes = [([], self.label_info(None, None))
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

    def label_info(self, patient, admission):  # TODO
        if patient is None:
            return (None, None)
        return (admission.patient_id, admission.admission_id)

    def label_pack(self, label_info):  # TODO
        '''Pack python-list label batches into numpy batches if needed'''
        return label_info


def main(_):
    '''Reader tests'''
    config = Config()
    vocab = NoteVocab(config)
    vocab.load_from_pickle()

    reader = NoteICD9Reader(config, vocab)
    words = 0
    for batch in reader.get(['train']):
        for note in batch[0]:
            words += len(note)
#            print(note)
#            for e in note:
#                print(vocab.vocab[e], end=' ')
#            print()
#            print()
    print(words)


if __name__ == '__main__':
    tf.app.run()
