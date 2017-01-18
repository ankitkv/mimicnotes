from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from multiprocessing import Pool
import re
import shelve

import nltk


re_anon = re.compile(r'\[\*\*.*?\*\*\]')
fix_re = re.compile(r"[^a-z0-9/'?.,-]+")
num_re = re.compile(r'[0-9]+')
dash_re = re.compile(r'-+')


def fix_word(word):
    word = word.lower()
    word = fix_re.sub('-', word)
    word = word.replace('-anon-', '<anon>')
    word = num_re.sub('#', word)
    word = dash_re.sub('-', word)
    return word.strip('-')


def mimic_tokenize(text):
    '''Takes in a raw string and returns a list of sentences, each sentence being a list of
       cleaned words.'''
    ret = []
    for sent in nltk.sent_tokenize(text):
        sent = re_anon.sub('-anon-', sent)
        words = nltk.word_tokenize(sent)
        words = [fix_word(word) for word in words]
        words = [word for word in words if word]
        ret.append(words)
    return ret


def partial_vocab(args):
    patients_list, shlf_file = args
    shelf = shelve.open(shlf_file)
    fd = nltk.FreqDist()
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
                for sent in mimic_tokenize(note.note_text):
                    fd.update(sent)
    return fd


def grouper(n, iterable, fillvalue=None):
    '''Group elements of iterable in groups of n. For example:
       >>> [e for e in grouper(3, [1,2,3,4,5,6,7])]
       [(1, 2, 3), (4, 5, 6), (7, None, None)]'''
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def mt_map(threads, func, operands):
    '''Multithreaded map if threads > 1. threads = 1 is useful for debugging.'''
    if threads > 1:
        p = Pool(threads)
        return p.map(func, operands)
    else:
        return map(func, operands)
