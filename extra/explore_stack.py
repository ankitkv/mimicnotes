from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from multiprocessing import Pool
import nltk
from pathlib import Path
import re
import shelve


data_dir = '../data/stack'
cpus = 7


re_anon = re.compile(r'\[\*\*.*?\*\*\]')
fix_re = re.compile(r"[^a-z0-9/?.,-:+#]+")
num_re = re.compile(r'[0-9]{2,}')
dash_re = re.compile(r'-+')


def fix_word(word, fix_anon=True):
    word = word.lower()
    word = fix_re.sub('-', word)
    if fix_anon:
        word = word.replace('-anon-', '<anon>')
    word = num_re.sub('#', word)
    word = dash_re.sub('-', word)
    return word.strip('-')


def mimic_tokenize(text, fix_anon=True):
    '''Takes in a raw string and returns a list of sentences, each sentence being a list of
       cleaned words.'''
    ret = []
    for sent in nltk.sent_tokenize(text):
        if fix_anon:
            sent = re_anon.sub('-anon-', sent)
        words = nltk.word_tokenize(sent)
        words = [fix_word(word, fix_anon) for word in words]
        words = [word for word in words if word]
        ret.append(words)
    return ret


def process_text(args):
    ids, shelve_file = args
    shelf = shelve.open(str(Path(data_dir) / 'questions.shelve'), 'r')
    vocab = nltk.FreqDist()
    tag_freq = nltk.FreqDist()
    lens = []
    prev_perc = -1
    print('Processing text...')
    for i, Id in enumerate(ids):
        row = shelf[Id]
        perc = int(100 * (i / len(ids)))
        if perc != prev_perc:
            if perc % 10 == 0:
                print(perc, i)
            prev_perc = perc
        title = row['Title']
        body = row['Body']
        tags = row['Tags']
        text = title + ' : ' + body
        tag_freq.update(tags)
        note_len = 0
        for sent in mimic_tokenize(text, fix_anon=False):
            note_len += len(sent)
            for word in sent:
                vocab[word] += 1
        lens.append(note_len)
    shelf.close()
    return vocab, tag_freq, lens


if __name__ == '__main__':
    shelve_file = str(Path(data_dir) / 'questions.shelve')
    shelf = shelve.open(shelve_file, 'r')
    ids = shelf.keys()
    shelf.close()
    tag_freq = nltk.FreqDist()
    vocab = nltk.FreqDist()
    lens = []
    group_size = int(0.999 + (len(ids) / cpus))
    grouped_ids = [ids[i:i+group_size] for i in range(0, len(ids), group_size)]
    p = Pool(cpus)
    ret = p.map_async(process_text, zip(grouped_ids, [shelve_file] * len(grouped_ids))).get(9999999)
    p.close()
    p.join()
    for part_vocab, part_tagfreq, part_lens in ret:
        vocab.update(part_vocab)
        tag_freq.update(part_tagfreq)
        lens.extend(part_lens)
    plt.hist(lens, 50, facecolor='green', alpha=0.75)
    plt.show()

    print('Total samples:', len(lens))

    print('Total tag mentions:', tag_freq.N())
    print('Tag vocab size:', tag_freq.B())
    i = 0
    cumsum = 0
    prev_perc = -1
    print('PercentCoverage Tags:')
    for w, c in tag_freq.most_common():
        i += 1
        cumsum += c
        perc = int((cumsum / tag_freq.N()) * 100)
        if perc != prev_perc:
            print(perc, i)
            prev_perc = perc

    print('Words:', vocab.N())
    print('Vocab size:', vocab.B())
    i = 0
    cumsum = 0
    prev_perc = -1
    print('PercentCoverage Words:')
    for w, c in vocab.most_common():
        i += 1
        cumsum += c
        perc = int((cumsum / vocab.N()) * 100)
        if perc != prev_perc:
            print(perc, i)
            prev_perc = perc
