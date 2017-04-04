from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import glob
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np


# True if need to find the best epoch instead of the last epoch
find_best_epoch = True


if __name__ == '__main__':
    fnames = glob.glob('../saved/*search.pk')
    data = []
    for fname in fnames:
        with open(fname, 'rb') as f:
            data.append(pickle.load(f))
    # now   data     = [(l1, records), ...] each for one value of l1
    # where records  = [stats, ...]         each for one epoch
    # where stats    = [prf_dict, ...]      each for one step
    # where prf_dict = {threshold: f_score}      for one step

    macro_data = []
    for l1, records in data:
        new_records = []
        for stats in records:
            stepf1s = collections.defaultdict(list)
            for prf_dict in stats:
                for k, v in prf_dict.items():
                    stepf1s[k].append(v)
            macro = {}
            for k, ls in stepf1s.items():
                macro[k] = np.mean(ls, axis=0)
            new_records.append(macro)
        macro_data.append((l1, new_records))
    # now   macro_data = [(l1, records), ...]  each for one value of l1
    # where records    = [prf_dict, ...]       each for one epoch
    # where prf_dict   = {threshold: f_scores}      for one epoch
    # can be used to find optimal threshold and optimal l1

    nothres_data = []
    for l1, records in macro_data:
        new_records = []
        for prf_dict in records:
            f_scores = np.amax(prf_dict.values(), axis=0)
            new_records.append(f_scores)
        nothres_data.append((l1, new_records))
    # now   nothres_data = [(l1, records), ...] each for one value of l1
    # where records      = [f_scores, ...]      each for one epoch

    l1_data = []
    argmaxes = {}
    if find_best_epoch:
        for l1, records in nothres_data:
            argmaxes[l1] = np.argmax(records, axis=0)
            l1_data.append((l1, np.amax(records, axis=0)))
    else:
        for l1, records in nothres_data:
            argmaxes[l1] = np.ones_like(records[0], dtype=np.int32) * (len(records) - 1)
            l1_data.append((l1, records[-1]))
    # now l1_data = [(l1, f_scores), ...] each for one value of l1

    l1s = np.array([l1 for l1, f_scores in l1_data])
    f_scores = [f_scores for l1, f_scores in l1_data]
    bests = np.argmax(f_scores, axis=0)
    final_l1s = l1s[bests]
    print('Best L1s:')
    print(final_l1s.tolist())
    print()

    thres_data = []
    for l1, records in macro_data:
        new_records = []
        for prf_dict in records:
            items = prf_dict.items()
            keys = np.array([k for k, v in items])
            values = [v for k, v in items]
            indices = np.argmax(values, axis=0)
            thresholds = keys[indices]
            new_records.append(thresholds)
        thres_data.append((l1, new_records))
    # now   thres_data = [(l1, records), ...] each for one value of l1
    # where records    = [thresholds, ...]    each for one epoch

    overall_data = {}
    for l1, records in thres_data:
        thresholds = np.zeros([len(final_l1s)], dtype=np.int32)
        for i, epoch_index in enumerate(argmaxes[l1]):
            thresholds[i] = records[epoch_index][i]
        overall_data[l1] = thresholds

    final_thresholds = []
    for i, l1 in enumerate(final_l1s):
        final_thresholds.append(overall_data[l1][i])
    final_thresholds = np.array(final_thresholds) / 10

    print('Best thresholds:')
    print(final_thresholds.tolist())
    print()

    if find_best_epoch:
        fname = '../saved/bow_bests_bestepoch.pk'
    else:
        fname = '../saved/bow_bests_lastepoch.pk'
    with open(fname, 'wb') as f:
        #pickle.dump([final_l1s, final_thresholds], f, -1)
        pickle.dump(final_l1s, f, -1)
    print('Dumped to pickle.')
