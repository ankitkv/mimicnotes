from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import nltk
import numpy as np
from pathlib import Path
import re

try:
    import cPickle as pickle
except ImportError:
    import pickle


data_dir = '../data/nyt2/'

fix_re = re.compile(r"[^a-z0-9/?.,-:]+")
num_re = re.compile(r'[0-9]{2,}')
dash_re = re.compile(r'-+')

cpus = 8


def fix_word(word):
    word = word.lower()
    word = fix_re.sub('-', word)
    word = num_re.sub('#', word)
    word = dash_re.sub('-', word)
    return word.strip('-')


def tokenize(text):
    ret = []
    for sent in nltk.sent_tokenize(text):
        words = nltk.word_tokenize(sent)
        words = [fix_word(word) for word in words]
        words = [word for word in words if word]
        ret.append(words)
    return ret


def process_text(rows):
    vocab = nltk.FreqDist()
    lens = []
    prev_perc = -1
    print('Processing text...')
    for i, row in enumerate(rows):
        perc = int(100 * (i / len(rows)))
        if perc != prev_perc:
            if perc % 10 == 0:
                print(perc, i)
            prev_perc = perc
        filename = row['Filename'][len('data/'):].replace('.xml', '.fulltext.txt')
        text_file = data_dir + 'text/data/' + filename
        with Path(text_file).open('r') as f:
            text = tokenize(f.read().encode('ascii', errors='ignore').lower())
        note_len = 0
        for sent in text:
            note_len += len(sent)
            for word in sent:
                vocab[word] += 1
        lens.append(min(note_len, 4500))
    return vocab, lens


if __name__ == '__main__':
    labels = nltk.FreqDist()
    with open(data_dir + 'meta.csv') as f:
        reader = csv.DictReader(f)
        rows = []
        for i, r in enumerate(reader):
            rows.append(r)
    print('Keys:', rows[0].keys())
    print('Total:', len(rows))
    min_labels = 99999
    max_labels = 0
    zeros = 0
    all_lens = []
    for row in rows:
        labs = row['Descriptors'].lower().split('|')
        min_labels = min(min_labels, len(labs))
        max_labels = max(max_labels, len(labs))
        all_lens.append(len(labs))
        labels.update(labs)
    print('Labels:', len(labels))
    print('Hapaxes:', len(labels.hapaxes()))
    print('Zeros:', zeros)
    print('Minimum:', min_labels)
    print('Maximum:', max_labels)
    print('Average:', np.mean(all_lens))
    #with open('labels.pk', 'wb') as f:
    #    pickle.dump(labels, f, -1)
    plt.hist(all_lens, 50, facecolor='green', alpha=0.75)
    plt.show()

    vocab = nltk.FreqDist()
    lens = []
    print('Processing text...')
    group_size = int(0.5 + (len(rows) / cpus))
    grouped_rows = [rows[i:i+group_size] for i in range(0, len(rows), group_size)]
    p = Pool(cpus)
    ret = p.map_async(process_text, grouped_rows).get(9999999)
    p.close()
    p.join()
    for part_vocab, part_lens in ret:
        vocab.update(part_vocab)
        lens.extend(part_lens)
    plt.hist(lens, 50, facecolor='green', alpha=0.75)
    plt.show()

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


#Max note len 2000 or 2500

#Words: 80778822
#Vocab size: 252942
#PercentCoverage Words:
#5 1
#11 2
#15 3
#17 4
#20 5
#22 6
#24 7
#26 8
#27 9
#28 10
#29 11
#30 12
#31 13
#32 14
#33 16
#34 17
#35 19
#36 21
#37 23
#38 25
#39 27
#40 30
#41 32
#42 35
#43 38
#44 42
#45 47
#46 52
#47 58
#48 65
#49 72
#50 79
#51 88
#52 97
#53 107
#54 118
#55 131
#56 146
#57 163
#58 180
#59 200
#60 222
#61 246
#62 273
#63 302
#64 335
#65 370
#66 409
#67 452
#68 498
#69 550
#70 606
#71 667
#72 733
#73 805
#74 885
#75 973
#76 1071
#77 1177
#78 1295
#79 1424
#80 1568
#81 1729
#82 1908
#83 2105
#84 2328
#85 2583
#86 2873
#87 3207
#88 3595
#89 4048
#90 4577
#91 5216
#92 6005
#93 7014
#94 8326
#95 10105
#96 12639
#97 16553
