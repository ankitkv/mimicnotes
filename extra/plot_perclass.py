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


index = 3  # zero-indexed from: [p, r, f, ap, auc, p8]
poly_degree = 0  # <= 0 to disable


if __name__ == '__main__':
    fnames = glob.glob('../saved/*.plot')
    data = []
    for fname in fnames:
        with open(fname, 'rb') as f:
            data.append(pickle.load(f))
    colors = cm.rainbow(np.linspace(0, 1, len(data)))
    for i, (label, perclass) in enumerate(data):
        plot_data = perclass[index]
        if poly_degree > 0:
            x = np.arange(len(plot_data))
            coefs = np.polyfit(x, plot_data, poly_degree)
            plot_data = np.polyval(coefs, x)
        plt.plot(plot_data, c=colors[i], label=label)
    plt.legend()
    plt.show()
