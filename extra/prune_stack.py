'''Prune the stack exchange dataset to contain questions with length >= 100 words.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bs4
import collections
import csv
from multiprocessing import Pool
from pathlib import Path
import re
import shelve


data_dir = '../data/stack/data'
out_dir = '../data/stack'
cpus = 7


alpha_re = re.compile(r"^[\sa-z0-9?.,-:]+$")


def process_text(rows):
    ret = []
    for i, row in enumerate(rows):
        body = row['Body']
        soup = bs4.BeautifulSoup(body, 'lxml')
        for code in soup.find_all('code'):
            code.clear()
        text = soup.text
        if i % 5000 == 0:
            print(i, i*100 / len(rows))
        length = len(text.split())
        if length < 100:  # skip questions shorter than 100 words
            continue
        row['Body'] = text
        ret.append(row)
    return ret


def clean(s):
    '''Crazy hack to normalize data'''
    news = []
    for c in s:
        if ord(c) < 128:
            news.append(c)
    return ''.join(news)


if __name__ == '__main__':
    print('Reading tags ...')
    tagset = collections.defaultdict(set)
    with (Path(data_dir) / 'Tags.csv').open('rb') as f:
        reader = csv.reader(f)
        reader.next()  # Id,Tag
        for row in reader:
            tagset[int(row[0])].add(row[1])

    print('Reading questions ...')
    with (Path(data_dir) / 'Questions.csv').open('rb') as f:
        reader = csv.DictReader(f)
        # Id,OwnerUserId,CreationDate,ClosedDate,Score,Title,Body
        rows = [r for r in reader]
    print('Pruning ...')
    group_size = int(0.999 + (len(rows) / cpus))
    grouped_rows = [rows[i:i+group_size] for i in range(0, len(rows), group_size)]
    p = Pool(cpus)
    ret = p.map_async(process_text, grouped_rows).get(9999999)
    p.close()
    p.join()
    rows = sum(ret, [])

    print('Writing ...')
    out_shelf = shelve.open(str(Path(out_dir) / 'questions.shelve'), 'c', protocol=-1,
                            writeback=True)
    for row in rows:
        row['Tags'] = list(tagset[int(row['Id'])])
        for k in ['Title', 'Body']:
            row[k] = clean(row[k])
        out_shelf[row['Id']] = row
