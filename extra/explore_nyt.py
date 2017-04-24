from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import matplotlib.pyplot as plt
import nltk
import numpy as np
import re

try:
    import cPickle as pickle
except ImportError:
    import pickle


data_dir = '../data/nyt/'

fix_re = re.compile(r"[^a-z0-9/?.,-:]+")
num_re = re.compile(r'[0-9]{2,}')
dash_re = re.compile(r'-+')


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
    words = 0
    prev_perc = -1
    print('Processing text...')
    for i, row in enumerate(rows):
        perc = int(100 * (i / len(rows)))
        if perc != prev_perc:
            print(perc, i)
            prev_perc = perc
        filename = row['Filename'][len('data/'):].replace('.xml', '.fulltext.txt')
        text_file = data_dir + 'text/data/' + filename
        with open(text_file, 'r') as f:
            text = tokenize(f.read().decode('ascii', errors='ignore').lower())
        for sent in text:
            for word in sent:
                vocab[word] += 1
                words += 1

    print('#words:', words)
    print('Vocab size:', vocab.B())
    i = 0
    cumsum = 0
    prev_perc = -1
    print('PercentCoverage Words:')
    for w, c in vocab.most_common():
        i += 1
        cumsum += c
        perc = int((cumsum / words) * 100)
        if perc != prev_perc:
            print(perc, i)
            prev_perc = perc
