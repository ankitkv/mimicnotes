from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
from pathlib import Path
import random
import shelve
from six.moves import xrange
try:
    import cPickle as pickle
except:
    import pickle

import nltk
import numpy as np
import tensorflow as tf

import util


class NoteData(object):
    '''Represents the tokenized note data'''

    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose
        self.patients_list = []

    def setup_splits(self):
        self.splits = {}
        trainidx = int(self.config.train_split * len(self.patients_list))
        validx = trainidx + int(self.config.val_split * len(self.patients_list))
        self.splits['train'] = self.patients_list[:trainidx]
        self.splits['val'] = self.patients_list[trainidx:validx]
        self.splits['test'] = self.patients_list[validx:]

    def get_patients_list(self, splits):
        return sum([self.splits[s] for s in splits], [])


class NoteShelveData(NoteData):
    '''Tokenized note data accessed via shelve files'''

    def __init__(self, config, max_cache_size=50000, verbose=True, load_from_pickle=True):
        super(NoteShelveData, self).__init__(config, verbose=verbose)
        if verbose:
            print('Using shelve data storage.')
        self.max_cache_size = max_cache_size
        nshelf_file = 'notes'
        if config.note_type:
            nshelf_file += '.' + config.note_type
        nshelf_file += '.shlf'
        self.nshelf_file = Path(config.data_path) / nshelf_file
        # cache admissions for faster loading in later epochs
        self.cache = collections.defaultdict(list)
        if load_from_pickle:
            self.load_from_pickle()

    def prepare_shelf(self, chunk_size=1024):
        if self.verbose:
            print('Preparing tokenized notes shelve from data...')
        if 'mimic2' in self.config.data_path:
            with (Path(self.config.data_path) / 'data/MIMIC_FILTERED_DSUMS').open('r') as f:
                note_data = [l.strip() for l in f]
            with (Path(self.config.data_path) / 'data/MIMIC_ICD9_codes').open('r') as f:
                labels = [l.strip() for l in f]
            with (Path(self.config.data_path) / 'data/MIMIC_ICD9_mapping').open('r') as f:
                icd9s = [l.strip().split('\t', 2) for l in f]
            assert len(note_data) == len(labels)
            icd9s = {index: (code, desc) for (code, index, desc) in icd9s}
            patients_list = [label.split('|', 1)[0] for label in labels]
        elif 'mimic' in self.config.data_path:
            pshelf_file = Path(self.config.data_path) / 'processed/patients.shlf'
            plist_file = Path(self.config.data_path) / 'processed/patients_list.pk'
            with plist_file.open('rb') as f:
                patients_list = pickle.load(f)
        elif 'nyt' in self.config.data_path:
            with (Path(self.config.data_path) / 'meta.csv').open('rb') as f:
                reader = csv.DictReader(f)
                pairs = [(r['Taxonomic Classifiers'], r['Filename']) for r in reader]
            patients_list = [str(i) for i in xrange(len(pairs))]
        nshelf = shelve.open(str(self.nshelf_file), 'c', protocol=-1, writeback=True)
        patients_set = set()
        for i in xrange(0, len(patients_list), chunk_size):
            plist = patients_list[i:i+chunk_size]
            if self.verbose:
                print('Chunk', i)
            group_size = int(0.5 + (len(plist) / self.config.threads))
            if 'mimic2' in self.config.data_path:
                partial_data = note_data[i:i+chunk_size]
                partial_labels = labels[i:i+chunk_size]
                lists = [(plist[j:j+group_size], partial_data[j:j+group_size],
                          partial_labels[j:j+group_size], icd9s)
                         for j in xrange(0, len(plist), group_size)]
                data = util.mt_map(self.config.threads, util.partial_tokenize_mimic2, lists)
            elif 'mimic' in self.config.data_path:
                lists = [plist[j:j+group_size] for j in xrange(0, len(plist), group_size)]
                data = util.mt_map(self.config.threads, util.partial_tokenize_mimic,
                                   zip(lists, [(str(pshelf_file),
                                                self.config.note_type)] * len(lists)))
            elif 'nyt' in self.config.data_path:
                rows = pairs[i:i+chunk_size]
                lists = [(plist[j:j+group_size], rows[j:j+group_size])
                         for j in xrange(0, len(plist), group_size)]
                data = util.mt_map(self.config.threads, util.partial_tokenize_nyt,
                                   zip(lists, [self.config.data_path] * len(lists)))
            for thread_data in data:
                for pid, (_, adm_map) in thread_data.items():
                    patients_set.add(pid)
                    nshelf[str(pid)] = adm_map
            nshelf.sync()
        nshelf.close()
        self.patients_list = []
        for pid in patients_list:
            if pid in patients_set:
                self.patients_list.append(pid)
        self.setup_splits()
        if self.verbose:
            print('Prepared.')

    def load_from_pickle(self):
        pat_list_file = 'notes_patients_list'
        if self.config.note_type:
            pat_list_file += '.' + self.config.note_type
        pat_list_file += '.pk'
        pat_list_file = Path(self.config.data_path) / pat_list_file
        try:
            if not self.nshelf_file.is_file():
                raise IOError
            with pat_list_file.open('rb') as f:
                self.patients_list = pickle.load(f)
            self.setup_splits()
        except IOError:
            self.prepare_shelf()
            with pat_list_file.open('wb') as f:
                pickle.dump(self.patients_list, f, -1)
        if self.verbose:
            print('Prepared to load data from shelve.')

    def iterate(self, splits=['train', 'val', 'test'], chunk_size=1536):
        '''Yields SimpleAdmission's from the data.'''
        patients_list = self.get_patients_list(splits)
        for i in xrange(0, len(patients_list), chunk_size):
            plist = []
            for pid in patients_list[i:i+chunk_size]:
                if pid in self.cache:
                    for admission in self.cache[pid]:
                        yield admission
                else:
                    plist.append(pid)
            if plist:
                group_size = int(0.5 + (len(plist) / self.config.threads))
                lists = [plist[j:j+group_size] for j in xrange(0, len(plist), group_size)]
                data = util.mt_map(self.config.threads, util.partial_read,
                                   zip(lists, [str(self.nshelf_file)] * len(lists)))
                for thread_data in data:
                    for admission in thread_data:
                        if len(self.cache) < self.max_cache_size:
                            self.cache[admission.patient_id].append(admission)
                        yield admission


class NotePickleData(NoteData):
    '''Tokenized note data accessed via pickle files'''

    def __init__(self, config, max_cache_size=50000, verbose=True, load_from_pickle=True):
        super(NotePickleData, self).__init__(config, verbose=verbose)
        if verbose:
            print('Using pickle data storage.')
        self.max_cache_size = max_cache_size
        self.bucket_map = {}
        notes_file = 'notes'
        if self.config.note_type:
            notes_file += '.' + self.config.note_type
        self.notes_file = notes_file + '.pk'
        # cache admissions for faster loading in later epochs
        self.cache = collections.defaultdict(list)
        if load_from_pickle:
            self.load_from_pickle()

    def prepare_pickles(self, chunk_size=1024, bucket_size=4096):
        if self.verbose:
            print('Preparing tokenized notes pickle from data...')
        if 'mimic2' in self.config.data_path:
            with (Path(self.config.data_path) / 'data/MIMIC_FILTERED_DSUMS').open('r') as f:
                note_data = [l.strip() for l in f]
            with (Path(self.config.data_path) / 'data/MIMIC_ICD9_codes').open('r') as f:
                labels = [l.strip() for l in f]
            with (Path(self.config.data_path) / 'data/MIMIC_ICD9_mapping').open('r') as f:
                icd9s = [l.strip().split('\t', 2) for l in f]
            assert len(note_data) == len(labels)
            icd9s = {index: (code, desc) for (code, index, desc) in icd9s}
            patients_list = [label.split('|', 1)[0] for label in labels]
        elif 'mimic' in self.config.data_path:
            pshelf_file = Path(self.config.data_path) / 'processed/patients.shlf'
            plist_file = Path(self.config.data_path) / 'processed/patients_list.pk'
            with plist_file.open('rb') as f:
                patients_list = pickle.load(f)
        elif 'nyt' in self.config.data_path:
            with (Path(self.config.data_path) / 'meta.csv').open('rb') as f:
                reader = csv.DictReader(f)
                pairs = [(r['Taxonomic Classifiers'], r['Filename']) for r in reader]
            patients_list = [str(i) for i in xrange(len(pairs))]
        patients_set = set()
        patients_dict = {}
        self.bucket_map = {}
        bucket = 0
        count = 0
        buckets_dir = Path(self.config.data_path) / 'buckets'
        try:
            buckets_dir.mkdir()
        except OSError:
            pass
        for i in xrange(0, len(patients_list), chunk_size):
            plist = patients_list[i:i+chunk_size]
            if self.verbose:
                print('Bucket', bucket, ' chunk', count)
            group_size = int(0.5 + (len(plist) / self.config.threads))
            if 'mimic2' in self.config.data_path:
                partial_data = note_data[i:i+chunk_size]
                partial_labels = labels[i:i+chunk_size]
                lists = [(plist[j:j+group_size], partial_data[j:j+group_size],
                          partial_labels[j:j+group_size], icd9s)
                         for j in xrange(0, len(plist), group_size)]
                data = util.mt_map(self.config.threads, util.partial_tokenize_mimic2, lists)
            elif 'mimic' in self.config.data_path:
                lists = [plist[j:j+group_size] for j in xrange(0, len(plist), group_size)]
                data = util.mt_map(self.config.threads, util.partial_tokenize_mimic,
                                   zip(lists, [(str(pshelf_file),
                                                self.config.note_type)] * len(lists)))
            elif 'nyt' in self.config.data_path:
                rows = pairs[i:i+chunk_size]
                lists = [(plist[j:j+group_size], rows[j:j+group_size])
                         for j in xrange(0, len(plist), group_size)]
                data = util.mt_map(self.config.threads, util.partial_tokenize_nyt,
                                   zip(lists, [self.config.data_path] * len(lists)))
            for thread_data in data:
                for pid, (patient, adm_map) in thread_data.items():
                    patients_set.add(pid)
                    self.bucket_map[pid] = bucket
                    patients_dict[pid] = (patient, adm_map)
            count += 1
            if count * chunk_size >= bucket_size:
                notes_file = buckets_dir / (self.notes_file + ('.%d' % bucket))
                with notes_file.open('wb') as f:
                    pickle.dump(patients_dict, f, -1)
                patients_dict = {}
                bucket += 1
                count = 0
        if patients_dict:
            notes_file = buckets_dir / (self.notes_file + ('.%d' % bucket))
            with notes_file.open('wb') as f:
                pickle.dump(patients_dict, f, -1)
        self.patients_list = []
        for pid in patients_list:
            if pid in patients_set:
                self.patients_list.append(pid)
        self.setup_splits()
        if self.verbose:
            print('Prepared.')

    def load_from_pickle(self):
        pat_list_file = 'notes_patients_list'
        if self.config.note_type:
            pat_list_file += '.' + self.config.note_type
        pat_list_file += '.pk'
        pat_list_file = Path(self.config.data_path) / pat_list_file
        bucket_file = 'bucket_map'
        if self.config.note_type:
            bucket_file += '.' + self.config.note_type
        bucket_file += '.pk'
        bucket_file = Path(self.config.data_path) / bucket_file
        try:
            notes_file = Path(self.config.data_path) / 'buckets' / (self.notes_file + '.0')
            if not notes_file.is_file():
                raise IOError
            with bucket_file.open('rb') as f:
                self.bucket_map = pickle.load(f)
            with pat_list_file.open('rb') as f:
                self.patients_list = pickle.load(f)
            self.setup_splits()
        except IOError:
            self.prepare_pickles()
            with bucket_file.open('wb') as f:
                pickle.dump(self.bucket_map, f, -1)
            with pat_list_file.open('wb') as f:
                pickle.dump(self.patients_list, f, -1)
        if self.verbose:
            print('Loaded data from pickle.')

    def iterate(self, splits=['train', 'val', 'test']):
        '''Yields SimpleAdmission's from the data.'''
        patients_list = self.get_patients_list(splits)
        buckets_dir = Path(self.config.data_path) / 'buckets'
        bucket = -1
        for pid in patients_list:
            if pid in self.cache:
                for admission in self.cache[pid]:
                    yield admission
            else:
                if bucket != self.bucket_map[pid]:
                    bucket = self.bucket_map[pid]
                    notes_file = buckets_dir / (self.notes_file + ('.%d' % bucket))
                    with notes_file.open('rb') as f:
                        patients_dict = pickle.load(f)
                _, adm_map = patients_dict[pid]
                for admission in adm_map.values():
                    if len(self.cache) < self.max_cache_size:
                        self.cache[admission.patient_id].append(admission)
                    yield admission


class NoteVocab(object):
    '''Stores the vocab: forward and reverse mappings, for text and other auxiliary info'''

    def __init__(self, config, data, verbose=True, load_from_pickle=True):
        self.config = config
        self.data = data
        self.verbose = verbose
        self.vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab_lookup = {w: i for i, w in enumerate(self.vocab)}
        self.aux_vocab = collections.defaultdict(list)
        self.aux_vocab_lookup = collections.defaultdict(dict)
        self.aux_names = collections.defaultdict(dict)
        self.sos_index = self.vocab_lookup.get('<sos>')
        self.eos_index = self.vocab_lookup.get('<eos>')
        self.unk_index = self.vocab_lookup.get('<unk>')
        if load_from_pickle:
            self.load_from_pickle()

    def vocab_freqs(self, notes_count):
        '''Lists the counts of each word in the vocabulary in the order of self.vocab'''
        if self.verbose:
            print('Loading frequencies for vocab...')
        vocab_fd_file = 'vocab_fd'
        if self.config.note_type:
            vocab_fd_file += '.' + self.config.note_type
        vocab_fd_file += '.pk'
        fdfile = Path(self.config.data_path) / vocab_fd_file
        with fdfile.open('rb') as f:
            vocab_fd, _ = pickle.load(f)
        ret = [vocab_fd[w] for w in self.vocab]
        ret[self.unk_index] = ((1.0 - self.config.keep_vocab) * sum(ret)) // self.config.keep_vocab
        ret[self.sos_index] = ret[self.eos_index] = notes_count
        if self.verbose:
            print('Loaded from freq dist file.')
        return ret

    def aux_vocab_freqs(self, key):
        '''Lists the counts of each value for a key in the aux vocabulary'''
        if self.verbose:
            print('Loading frequencies for %s...' % key)
        vocab_fd_file = 'vocab_fd'
        if self.config.note_type:
            vocab_fd_file += '.' + self.config.note_type
        vocab_fd_file += '.pk'
        fdfile = Path(self.config.data_path) / vocab_fd_file
        with fdfile.open('rb') as f:
            _, vocab_aux_fd = pickle.load(f)
        ret = [vocab_aux_fd[key][(w, self.aux_names[key][w])] for w in self.aux_vocab[key]]
        if self.verbose:
            print('Loaded from aux freq dist file.')
        return ret

    def prepare_vocab_from_fd(self, vocab_fd, vocab_aux_fd):
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
        if self.verbose:
            print('Pruned vocabulary loaded, size:', len(self.vocab))
            for k, v in self.aux_vocab.items():
                print(k + ':', len(v))

    def load_by_parsing(self):
        '''Read the vocab from the dataset'''
        if self.verbose:
            print('Loading vocabulary by parsing...')
        vocab_fd = nltk.FreqDist()
        vocab_aux_fd = collections.defaultdict(nltk.FreqDist)
        for adm in self.data.iterate():
            for note in adm.notes:
                for sent in note:
                    vocab_fd.update(sent)
            for pres in adm.psc_events:
                vocab_aux_fd['psc'].update([pres])
            for proc in adm.pcd_events:
                vocab_aux_fd['pcd'].update([proc])
            for diag in adm.dgn_events:
                vocab_aux_fd['dgn'].update([diag])
        if self.verbose:
            print('Full vocabulary size:', vocab_fd.B())
        self.prepare_vocab_from_fd(vocab_fd, vocab_aux_fd)
        return vocab_fd, vocab_aux_fd

    def load_from_pickle(self):
        '''Read the vocab from pickled files, saving if necessary'''
        vocab_file = 'vocab'
        if self.config.note_type:
            vocab_file += '.' + self.config.note_type
        vocab_file += '.%.2f.pk' % self.config.keep_vocab
        pkfile = Path(self.config.data_path) / vocab_file
        try:
            if self.verbose:
                print('Loading vocabulary from pickle...')
            with pkfile.open('rb') as f:
                self.vocab, self.vocab_lookup, \
                            self.aux_vocab, self.aux_vocab_lookup, self.aux_names = pickle.load(f)
            if self.verbose:
                print('Vocabulary loaded, size:', len(self.vocab))
                for k, v in self.aux_vocab.items():
                    print(k + ':', len(v))
        except IOError:
            if self.verbose:
                print('Error loading from pickle, processing from freq dist for new keep_vocab.')
            vocab_fd_file = 'vocab_fd'
            if self.config.note_type:
                vocab_fd_file += '.' + self.config.note_type
            vocab_fd_file += '.pk'
            fdfile = Path(self.config.data_path) / vocab_fd_file
            try:
                if self.verbose:
                    print('Loading vocab freq dist from pickle...')
                with fdfile.open('rb') as f:
                    vocab_fd, vocab_aux_fd = pickle.load(f)
                if self.verbose:
                    print('Full vocabulary loaded, size:', vocab_fd.B())
                self.prepare_vocab_from_fd(vocab_fd, vocab_aux_fd)
            except IOError:
                if self.verbose:
                    print('Error loading freq dist from pickle, attempting parsing.')
                vocab_fd, vocab_aux_fd = self.load_by_parsing()
                with fdfile.open('wb') as f:
                    pickle.dump([vocab_fd, vocab_aux_fd], f, -1)
                    if self.verbose:
                        print('Saved vocab freq dist pickle file.')
            with pkfile.open('wb') as f:
                pickle.dump([self.vocab, self.vocab_lookup, self.aux_vocab, self.aux_vocab_lookup,
                             self.aux_names], f, -1)
                if self.verbose:
                    print('Saved vocab pickle file.')

    def words2idxs(self, words):
        return [self.vocab_lookup.get(w, self.unk_index) for w in words]

    def idxs2words(self, idxs):
        return [self.vocab[idx] for idx in idxs]


class NoteReader(object):
    '''The reader that yields vectorized notes and their respective patient and admission IDs
       according to the requested split(s) (train, val, test)'''

    def __init__(self, config, data, vocab):
        self.config = config
        self.data = data
        self.vocab = vocab
        if config.curriculum:
            # length is incremented at the start of an epoch, so go back a bit to start correctly
            self.max_note_len = float(config.len_start) / (1 + config.len_multiply)
        else:
            self.max_note_len = float(config.max_note_len)
        random.seed(0)  # deterministic random

    def label_info(self, admission):
        '''Can be extended to provide different kinds of labels from a SimpleAdmission'''
        if admission is None:
            return (None, None)
        return (admission.patient_id, admission.admission_id)

    def label_space_size(self):
        return 2

    def read_notes(self, splits):
        '''Read single notes from data'''
        for adm in self.data.iterate(splits):
            label_info = self.label_info(adm)
            for note in adm.notes:
                note_text = sum(note, [])
                vocab_note = [self.vocab.sos_index] + self.vocab.words2idxs(note_text) + \
                             [self.vocab.eos_index]
                if self.max_note_len > 0:
                    vocab_note = vocab_note[:int(self.max_note_len)]
                yield (vocab_note, label_info)

    def buffered_read_sorted_notes(self, splits, batches=32):
        '''Read and return a list of notes (length multiple of batch_size) worth at most $batches
           number of batches sorted in length'''
        buffer_size = self.config.batch_size * batches
        notes = []
        for note in self.read_notes(splits):
            notes.append(note)
            if len(notes) == buffer_size:
                if self.config.length_sort:
                    notes.sort(key=lambda x: len(x[0]))
                else:
                    random.shuffle(notes)
                yield notes
                notes = []
        if notes:
            if self.config.length_sort:
                notes.sort(key=lambda x: len(x[0]))
            else:
                random.shuffle(notes)
            mod = len(notes) % self.config.batch_size
            if mod != 0:
                notes.extend([([], self.label_info(None))
                              for _ in xrange(self.config.batch_size - mod)])
            yield notes

    def buffered_read(self, splits):
        '''Read packed batches from data with each batch having notes of similar lengths'''
        for note_collection in self.buffered_read_sorted_notes(splits):
            batches = [note_collection[i:i+self.config.batch_size]
                       for i in xrange(0, len(note_collection), self.config.batch_size)]
            if self.config.length_sort:
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

    def get(self, splits, force_curriculum=None, verbose=True):
        '''Read batches from data'''
        if verbose:
            print('Getting data from', '+'.join(splits), 'split')
        if force_curriculum is None:
            force_curriculum = 'train' in splits
        if self.config.curriculum and force_curriculum:
            old = int(self.max_note_len)
            self.max_note_len += self.max_note_len * self.config.len_multiply
            if self.max_note_len > self.config.max_note_len:
                self.config.curriculum = False
                self.max_note_len = self.config.max_note_len
            new = int(self.max_note_len)
            if verbose and old != new:
                print('(reader) Increasing max note len to', new)
        for batch in self.buffered_read(splits):
            yield batch


class NoteICD9Reader(NoteReader):
    '''A note reader that considers ICD9 codes as labels'''

    def __init__(self, config, data, vocab):
        super(NoteICD9Reader, self).__init__(config, data, vocab)
        self.max_dgn_labels = len(self.vocab.aux_vocab['dgn'])
        if self.config.max_dgn_labels > 0:
            self.max_dgn_labels = min(self.max_dgn_labels, self.config.max_dgn_labels)

    def label_info(self, admission):
        label = np.zeros([self.max_dgn_labels], dtype=np.int)
        if admission is None:
            return label
        vocab_lookup = self.vocab.aux_vocab_lookup['dgn']
        for diag_code, _ in admission.dgn_events:
            try:
                label[vocab_lookup[diag_code]] = 1
            except IndexError:
                pass
        return label

    def label_space_size(self):
        return self.max_dgn_labels

    def label_pack(self, label_info):
        return np.array(label_info)


def main(_):
    '''Reader tests'''
    config = util.Config()
    if config.data_storage == 'shelve':
        data = NoteShelveData(config)
    elif config.data_storage == 'pickle':
        data = NotePickleData(config)
    vocab = NoteVocab(config, data)
    reader = NoteICD9Reader(config, data, vocab)
#    for k, v in vocab.aux_names['dgn'].items():
#        if v == 'Depressive disorder NEC':
#            target = vocab.aux_vocab_lookup['dgn'][k]
    for epoch in xrange(1):
        words = 0
        print('Epoch', epoch)
        for batch in reader.get(['val']):
            for i in xrange(batch[0].shape[0]):
                note = batch[0][i]
                words += len(note)
#                label = batch[2][i]
#                print_this = False
#                if label[target]:
#                    for e in note:
#                        if vocab.vocab[e] == 'anxiety':
#                            print_this = True
#                            break
#                if print_this:
#                    for e in note:
#                        print(vocab.vocab[e], end=' ')
#                    print('--------------')
#                    print()
#        print()
        print(words)


if __name__ == '__main__':
    tf.app.run()
