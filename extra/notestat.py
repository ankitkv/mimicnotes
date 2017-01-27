from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import itertools
import multiprocessing
from multiprocessing import Pool

import shelve
try:
    import cPickle as pickle
except:
    import pickle


def grouper(n, iterable, fillvalue=None):
    '''Group elements of iterable in groups of n. For example:
       >>> [e for e in grouper(3, [1,2,3,4,5,6,7])]
       [(1, 2, 3), (4, 5, 6), (7, None, None)]'''
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


with open('../data/processed/patients_list.pk') as f:
    patients_list = pickle.load(f)
# patients_list = patients_list[:1000]


def partial_stat(patients):
    shelf = shelve.open('../data/processed/patients.shlf')
    pat_notes = 0
    pat_no_notes = 0
    adm_notes = 0
    adm_no_notes = 0
    cat_count = nltk.FreqDist()
    for pid in patients:
        if pid is None:
            break
        try:
            int(pid)
        except ValueError:
            continue
        patient = shelf[pid]
        has_notes = False
        for adm in patient.admissions.values():
            if adm.nte_events:
                has_notes = True
                adm_notes += 1
            else:
                adm_no_notes += 1
            for note in adm.nte_events:
                cat_count.update([note.note_cat])
        if has_notes:
            pat_notes += 1
        else:
            pat_no_notes += 1
    print('Done')
    return (pat_notes, pat_no_notes, adm_notes, adm_no_notes, cat_count)


cores = int(.5 + (.9 * float(multiprocessing.cpu_count())))
print('Number of procs:', cores)
splits = int(.5 + (len(patients_list) / cores))

p = Pool(cores)
outs = p.map(partial_stat, grouper(splits, patients_list))
print('Collecting results')

pat_notes = 0
pat_no_notes = 0
adm_notes = 0
adm_no_notes = 0
cat_count = nltk.FreqDist()

for pn, pnn, an, ann, cc in outs:
    pat_notes += pn
    pat_no_notes += pnn
    adm_notes += an
    adm_no_notes += ann
    cat_count.update(cc)


print('Patients with notes:', pat_notes)
print('Patients with no notes:', pat_no_notes)
print('Admissions with notes:', adm_notes)
print('Admissions with no notes:', adm_no_notes)
print('Category counts:')
for k, v in cat_count.items():
    print(k, v)
