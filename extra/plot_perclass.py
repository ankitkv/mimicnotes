from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
try:
    import cPickle as pickle
except:
    import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np


index = 4  # zero-indexed from: [p, r, f, ap, auc, p8, r8]
poly_degree = -1  # <= 0 to disable
top = 1000  # look at the top these many concepts
conv_window = -1  # <= 1 to disable
batch_size = 50  # <= 0 to disable


if __name__ == '__main__':
    #fnames = glob.glob('../saved/*.plot')
    fnames = [
#        '../saved/bow200.plot',
#        '../saved/bow500.plot',
        ('../saved/bow1000.plot', 'Bag of words'),
#        '../saved/attn200_w3.plot',
#        '../saved/attn500_w3.plot',
        ('../saved/attn1000_w3.plot', 'Attention bag of words'),
#        '../saved/rnn2_f200_h128e192.plot',
#        '../saved/rnn2_f500_h128e192.plot',
#        '../saved/grnn_f200.plot',
#        '../saved/grnn_f500.plot',
#        '../saved/grnn_f1000.plot',
#        '../saved/grnn_f200_h0.plot',
#        '../saved/grnn_f500_h0.plot',
#        '../saved/grnn_f1000_h0.plot',
#        '../saved/grnnsd_f200_h128.plot',
#        '../saved/grnnsd_f500_h128.plot',
        ('../saved/grnnsd_f1000_h128.plot', 'Grounded RNN'),
#        '../saved/grnnsd_f1000_h128r1e-2.plot',
#        '../saved/grnnsd_f1000_h128r1e-3.plot',
#        '../saved/grnnsd_f1000_h128r1e-4.plot',
#        '../saved/grnnsd_f1000_h128r1e-5.plot',
    ]
    data = []
    for fname, name in fnames:
        with open(fname, 'rb') as f:
            data.append((name, pickle.load(f)))
    colors = cm.rainbow(np.linspace(0, 1, len(data)))
    for i, (label, perclass_tuple) in enumerate(data):
        _, perclass = perclass_tuple
        plot_data = perclass[index][:top]
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
                new_plot.append(plot_data[j: j + batch_size].mean())
            plot_data = np.array(new_plot)
        plt.plot(plot_data, c=colors[i], label=label)
    plt.legend()
    plt.xlabel('Group of size 50 of labels in the order of decreasing frequency')
    plt.ylabel('Area under the ROC curve')
    plt.show()
