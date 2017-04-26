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


data_dir = '../data/nyt/'

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
    labels = set()
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
        labs = row['Taxonomic Classifiers'].split('|')
        labs = [l for l in labs if l.startswith('Top')]
        if not len(labs):
            zeros += 1
        min_labels = min(min_labels, len(labs))
        max_labels = max(max_labels, len(labs))
        all_lens.append(len(labs))
        labels.update(labs)
    print('Labels:', len(labels))
    print('Zeros:', zeros)
    print('Minimum:', min_labels)
    print('Maximum:', max_labels)
    print('Average:', np.mean(all_lens))
    #with open('labels.pk', 'wb') as f:
    #    pickle.dump(labels, f, -1)
    #plt.hist(all_lens, 50, facecolor='green', alpha=0.75)
    #plt.show()

    vocab = nltk.FreqDist()
    lens = []
    print('Processing text...')
    group_size = int(0.5 + (len(rows) / cpus))
    grouped_rows = [rows[i:i+group_size] for i in range(0, len(rows), group_size)]
    p = Pool(cpus)
    for part_vocab, part_lens in p.map(process_text, grouped_rows):
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


# max length: 2000 or 1500


#Keys: ['Byline', 'Headline', 'Dateline', 'Online Titles', 'Filename', 'Taxonomic Classifiers', 'General Online Descriptors', 'Online Organizations', 'Word Count', 'Body', 'News Desk', 'Online Section', 'Publication Day Of Month', 'Online Headline', 'Descriptors', 'Day Of Week', 'Publication Year', 'Correction Date', 'Alternate URL', 'Kicker', 'Banner', 'Section', 'Online Descriptors', 'Organizations', 'Column Number', 'Feature Page', 'Url', 'Publication Date', 'Online People', 'Article Abstract', 'Credit', 'Types Of Material', 'Page', 'Lead Paragraph', 'Series Name', 'Biographical Categories', 'Correction Text', 'Online Locations', 'Publication Month', 'People', 'Locations', 'Slug', 'Normalized Byline', 'Online Lead Paragraph', 'Titles', 'Column Name', 'Names', 'Guid', 'Author Biography']
#Total: 402606
#Labels: 2518
#Zeros: 2194
#Minimum: 0
#Maximum: 84
#Average: 5.01711350551

#Words: 265076207
#Vocab size: 836510
#PercentCoverage Words:
#6 1
#11 2
#15 3
#18 4
#20 5
#22 6
#24 7
#26 8
#27 9
#28 10
#29 11
#30 12
#31 13
#32 15
#33 16
#34 18
#35 19
#36 21
#37 23
#38 26
#39 28
#40 31
#41 34
#42 37
#43 41
#44 45
#45 50
#46 55
#47 62
#48 69
#49 76
#50 85
#51 95
#52 108
#53 121
#54 138
#55 156
#56 177
#57 201
#58 226
#59 255
#60 286
#61 321
#62 359
#63 402
#64 450
#65 503
#66 560
#67 623
#68 691
#69 767
#70 850
#71 941
#72 1041
#73 1151
#74 1271
#75 1404
#76 1552
#77 1718
#78 1903
#79 2110
#80 2340
#81 2598
#82 2886
#83 3212
#84 3585
#85 4013
#86 4506
#87 5081
#88 5760
#89 6570
#90 7543
#91 8738
#92 10240
#93 12158
#94 14667
#95 18096
#96 23059
#97 30964

