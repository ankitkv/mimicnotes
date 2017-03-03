from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from multiprocessing import Pool
import re
import six
import shelve

import nltk
import numpy as np
import sklearn
import tensorflow as tf


class c:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


re_anon = re.compile(r'\[\*\*.*?\*\*\]')
fix_re = re.compile(r"[^a-z0-9/'?.,-:]+")
num_re = re.compile(r'[0-9]{2,}')
dash_re = re.compile(r'-+')


def fix_word(word):
    word = word.lower()
    word = fix_re.sub('-', word)
    word = word.replace('-anon-', '<anon>')
    word = num_re.sub('#', word)
    word = dash_re.sub('-', word)
    return word.strip('-')


def mimic_tokenize(text):
    '''Takes in a raw string and returns a list of sentences, each sentence being a list of
       cleaned words.'''
    ret = []
    for sent in nltk.sent_tokenize(text):
        sent = re_anon.sub('-anon-', sent)
        words = nltk.word_tokenize(sent)
        words = [fix_word(word) for word in words]
        words = [word for word in words if word]
        ret.append(words)
    return ret


class SimplePatient(object):
    '''A light-weight representation of a patient, built from a MimicPatient.'''

    def __init__(self, patient):
        self.patient_id = patient.patient_id
        self.gender = patient.gender


class SimpleAdmission(object):
    '''A light-weight representation of an admission, built from a MimicAdmission.'''

    def __init__(self, admission, tokenized_notes):
        self.patient_id = admission.patient_id
        self.admission_id = admission.admission_id
        self.adm_type = admission.adm_type
        self.psc_events = []
        for pres in admission.psc_events:
            ndc = pres.drug_codes[-1]
            if ndc == '0':
                name = '<missing>'
            else:
                name = pres.drug_names[0]
            self.psc_events.append((ndc, name))
        self.pcd_events = []
        for proc in admission.pcd_events:
            self.pcd_events.append((proc.code, proc.name))
        self.dgn_events = []
        for diag in admission.dgn_events:
            self.dgn_events.append((diag.code, diag.name))
        self.notes = tokenized_notes


def partial_tokenize(args):
    patients_list, (shlf_file, note_type) = args
    if note_type:
        note_type = note_type.replace('_', ' ')
    shelf = shelve.open(shlf_file, 'r')
    ret = {}
    for pid in patients_list:
        try:
            int(pid)
        except ValueError:
            continue
        patient = shelf[pid]
        found = False
        adm_map = {}
        for adm in patient.admissions.values():
            try:
                int(adm.admission_id)
            except ValueError:
                continue
            notes_list = []
            for note in adm.nte_events:
                if not note_type or note.note_cat == note_type:
                    note_text = []
                    for sent in mimic_tokenize(note.note_text):
                        note_text.append(sent)
                    notes_list.append(note_text)
            if notes_list:
                found = True
                adm_map[adm.admission_id] = SimpleAdmission(adm, notes_list)
        if found:
            ret[pid] = (SimplePatient(patient), adm_map)
    shelf.close()
    return ret


def partial_read(args):
    patients_list, nshlf_file = args
    nshelf = shelve.open(nshlf_file, 'r')
    ret = []
    for pid in patients_list:
        ret.extend(nshelf[pid].values())
    nshelf.close()
    return ret


def mt_map(threads, func, operands):
    '''Multithreaded map if threads > 1. threads = 1 is useful for debugging.'''
    if threads > 1:
        p = Pool(threads)
        ret = p.map_async(func, operands).get(9999999)
        p.close()
        p.join()
    else:
        ret = map(func, operands)
    return ret


def f1_score(probs, labels, thres, average='micro'):
    '''Returns (precision, recall, F1 score) from a batch of predictions (thresholded probabilities)
       given a batch of labels (for macro-averaging across batches)'''
    preds = (probs >= thres).astype(np.int32)
    p, r, f, _ = sklearn.metrics.precision_recall_fscore_support(labels, preds, average=average,
                                                                 warn_for=())
    return p, r, f


def average_precision(probs, labels):
    '''Precision integrated over all thresholds (area under the precision-recall curve)'''
    return sklearn.metrics.average_precision_score(labels, probs, average='micro')


def precision_at_k(probs, labels, k):
    indices = np.argpartition(-probs, k-1, axis=1)[:, :k]
    preds = np.zeros(probs.shape, dtype=np.int)
    preds[np.arange(preds.shape[0])[:, np.newaxis], indices] = 1
    return sklearn.metrics.precision_score(labels, preds, average='micro')


def torch_initialize(m):
    '''Initialize weights of a torch module'''
    import torch.nn as nn

    def linear_init(weight):
        fan_out, fan_in = weight.size()
        weight.data.normal_(0.0, np.sqrt(2.0 / (fan_in + fan_out)))
    if isinstance(m, nn.Linear):
        linear_init(m.weight)
    elif isinstance(m, nn.GRUCell):
        linear_init(m.weight_ih)
        linear_init(m.weight_hh)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.size()) == 2:
                linear_init(param)
                linear_init(param)
    elif isinstance(m, nn.Embedding):
        m.weight.data.uniform_()


def torch_optimizer(name, lr, params):
    '''Return an optimizer for Torch that learns params'''
    import torch.optim as optim
    if name == 'sgd':
        return optim.SGD(params, lr)
    elif name == 'adam':
        return optim.Adam(params, lr)
    elif name == 'adagrad':
        return optim.Adagrad(params, lr)
    elif name == 'adadelta':
        return optim.Adadelta(params, lr)


def prelu(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.

    Based on https://github.com/jimfleming/recurrent-entity-networks.
    """
    if initializer is None:
        initializer = tf.ones_initializer()
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg


def linear(args, output_size, bias=True, bias_start=0.0, scope=None, initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    Based on the code from TensorFlow."""
    if not isinstance(args, collections.Sequence) or isinstance(args, six.string_types):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size], dtype=dtype,
                                 initializer=initializer)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size], dtype=dtype,
                                    initializer=tf.constant_initializer(bias_start,
                                                                        dtype=dtype))
    return res + bias_term


def conv1d(inputs, output_dims, kernel_width, stride=1, padding='SAME', scope=None):
    '''Convolve one-dimensional data such as text.'''
    with tf.variable_scope(scope or "Convolution"):
        W_conv = tf.get_variable('W_conv', [kernel_width, inputs.get_shape()[-1].value,
                                            output_dims],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv = tf.get_variable('b_conv', [output_dims],
                                 initializer=tf.constant_initializer(0.0))
        conv_out = tf.nn.conv1d(inputs, W_conv, stride, padding)
        conv_out = tf.nn.bias_add(conv_out, b_conv)
    return conv_out
