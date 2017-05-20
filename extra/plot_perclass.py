from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except:
    import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np


index = 3  # zero-indexed from: [p, r, f, ap, auc]
poly_degree = -1  # <= 0 to disable
top = 10000  # look at the top these many concepts
conv_window = -1  # <= 1 to disable
batch_size = 200  # <= 0 to disable


if __name__ == '__main__':
    fnames = [
        ('../saved/m2_bow.plot',           'Logistic',      'b--^'),
        ('../saved/m2_attn_w3.plot',       'Attention BOW', 'b-^'),
        ('../saved/m2_rnn_h128.plot',      'GRU',           'g--v'),
        ('../saved/m2_rnn_h64_m0b1.plot',  'BiGRU',         'g-v'),
        ('../saved/m2_grnn_h128.plot',     'GRNN',          'r--o'),
        ('../saved/m2_grnn_h64_b1r1.plot', 'BiGRNN',        'r-o'),
    ]
    data = []
    for fname, name, style in fnames:
        with open(fname, 'rb') as f:
            data.append((name, style, pickle.load(f)))
    colors = cm.rainbow(np.linspace(0, 1, len(data)))
    datas = []
    for i, (label, _, perclass_tuple) in enumerate(data):
        _, perclass = perclass_tuple
        datas.append(perclass[3][:top])
    datas = np.mean(np.array(datas), 0)
    indices = np.isfinite(datas)
    plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size': 14})
    for i, (label, style, perclass_tuple) in enumerate(data):
        _, perclass = perclass_tuple
        plot_data = perclass[index][:top][indices]
        if poly_degree > 0:
            x = np.arange(len(plot_data))
            coefs = np.polyfit(x, plot_data, poly_degree)
            plot_data = np.polyval(coefs, x)
        if conv_window > 1:
            conv_side = conv_window // 2
            window = 1 + (conv_side * 2)
            start_pad = np.flip(plot_data[:conv_side], 0)
            end_pad = np.flip(plot_data[-conv_side:], 0)
            padded = np.concatenate([start_pad, plot_data, end_pad])
            plot_data = np.convolve(padded, np.ones([window]) / window, mode='valid')
        if batch_size > 0:
            new_plot = []
            for j in range(0, len(plot_data), batch_size):
                value = np.nanmean(plot_data[j: j + batch_size])
                new_plot.append(value)
            plot_data = np.array(new_plot)
        plt.plot(plot_data, style, label=label)
    plt.legend()
    plt.xlabel('Decreasing label frequency')
    if index == 0:
        plt.ylabel('Precision')
    elif index == 1:
        plt.ylabel('Recall')
    elif index == 2:
        plt.ylabel('F1-score')
    elif index == 3:
        plt.ylabel('Area under the PR curve')
    elif index == 4:
        plt.ylabel('Area under the ROC curve')
    #plt.show()
    plt.savefig('perclass_auc.pdf')
