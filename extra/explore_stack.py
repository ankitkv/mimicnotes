from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bs4
import collections
import csv
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import nltk
from pathlib import Path

try:
    import cPickle as pickle
except ImportError:
    import pickle


data_dir = '../data/stack/data'
cpus = 7


def process_text(rows):
    lens = []
    for i, row in enumerate(rows):
#        if row[-4] != 'NA':
#            continue  # ignore closed
        qid = int(row[0])
        body = row[-1]
        soup = bs4.BeautifulSoup(body, 'lxml')
        for code in soup.find_all('code'):
            code.clear()
        text = soup.text
        if i % 5000 == 0:
            print(i, i*100 / len(rows))
        length = len(text.split())
        if length < 100:  # skip questions shorter than 100 words
            continue
        lens.append(length)
#        print(text)
#        print('\n\n')
#        title = row[-2]
#        soup = bs4.BeautifulSoup(body, 'lxml')
#        tags = tagset[qid]
#        raw_input()
    return lens



if __name__ == '__main__':
    print('Reading tags ...')
    tagset = collections.defaultdict(set)
    tag_freq = nltk.FreqDist()
    with (Path(data_dir) / 'Tags.csv').open('rb') as f:
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
    with (Path(data_dir) / 'Questions.csv').open('rb') as f:
        reader = csv.reader(f)
        reader.next()  # Id,OwnerUserId,CreationDate,ClosedDate,Score,Title,Body
        rows = [r for r in reader]
    print('Analyzing ...')
    group_size = int(0.999 + (len(rows) / cpus))
    grouped_rows = [rows[i:i+group_size] for i in range(0, len(rows), group_size)]
    p = Pool(cpus)
    ret = p.map_async(process_text, grouped_rows).get(9999999)
#    ret = map(process_text, grouped_rows)
    p.close()
    p.join()

    lens = sum(ret, [])
    print(len(lens))
    plt.hist(lens, 50, facecolor='green', alpha=0.75)
    plt.show()

# use top 4k labels
