from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool
import nltk
from pathlib import Path
import re


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
        title = row[-2]
        body = row[-1]
        text = title + ' : ' + body
        note_len = 0
        for sent in mimic_tokenize(text, fix_anon=False):
            note_len += len(sent)
            for word in sent:
                vocab[word] += 1
        lens.append(note_len)
    return vocab, lens


if __name__ == '__main__':
    print('Reading tags ...')
    tagset = collections.defaultdict(set)
    tag_freq = nltk.FreqDist()
    with (Path(data_dir) / 'data/Tags.csv').open('rb') as f:
        reader = csv.reader(f)
        reader.next()  # Id,Tag
        for row in reader:
            tagset[int(row[0])].add(row[1])
            tag_freq.update([row[1]])
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

    print('Reading questions ...')
    with (Path(data_dir) / 'PrunedQuestions.csv').open('rb') as f:
        reader = csv.reader(f)
        reader.next()  # Id,OwnerUserId,CreationDate,ClosedDate,Score,Title,Body
        rows = [r for r in reader]
    print('Analyzing ...')
    vocab = nltk.FreqDist()
    lens = []
    group_size = int(0.999 + (len(rows) / cpus))
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


#Max note len 600

#Words: 77292315
#Vocab size: 832282
#PercentCoverage Words:
#5 1
#8 2
#12 3
#15 4
#18 5
#20 6
#22 7
#24 8
#26 9
#27 10
#29 11
#30 12
#31 13
#32 14
#33 15
#34 16
#35 18
#36 19
#37 20
#38 22
#39 24
#40 26
#41 28
#42 30
#43 33
#44 35
#45 38
#46 41
#47 44
#48 47
#49 51
#50 54
#51 59
#52 63
#53 68
#54 74
#55 81
#56 87
#57 95
#58 102
#59 110
#60 119
#61 129
#62 140
#63 152
#64 165
#65 180
#66 195
#67 212
#68 231
#69 252
#70 274
#71 298
#72 323
#73 352
#74 383
#75 417
#76 456
#77 501
#78 549
#79 603
#80 662
#81 729
#82 805
#83 890
#84 988
#85 1097
#86 1223
#87 1373
#88 1550
#89 1768
#90 2038
#91 2381
#92 2829
#93 3437
#94 4321
#95 5716
#96 8318
#97 14305

