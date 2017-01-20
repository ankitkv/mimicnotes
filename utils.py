from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
from multiprocessing import Pool
import re
import shelve

import nltk
import tensorflow as tf


re_anon = re.compile(r'\[\*\*.*?\*\*\]')
fix_re = re.compile(r"[^a-z0-9/'?.,-]+")
num_re = re.compile(r'[0-9]+')
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


def partial_vocab(args):
    patients_list, shlf_file = args
    shelf = shelve.open(shlf_file)
    fd = nltk.FreqDist()
    aux_fd = collections.defaultdict(nltk.FreqDist)
    for pid in patients_list:
        if pid is None:
            break
        try:
            int(pid)
        except ValueError:
            continue
        patient = shelf[pid]
        for adm in patient.admissions.values():
            for pres in adm.psc_events:
                ndc = pres.drug_codes[-1]
                if ndc == '0':
                    name = '<missing>'
                else:
                    name = pres.drug_names[0]
                aux_fd['psc'].update([(ndc, name)])
            for proc in adm.pcd_events:
                aux_fd['pcd'].update([(proc.code, proc.name)])
            for diag in adm.dgn_events:
                aux_fd['dgn'].update([(diag.code, diag.name)])
            for note in adm.nte_events:
                for sent in mimic_tokenize(note.note_text):
                    fd.update(sent)
    return fd, aux_fd


def grouper(n, iterable, fillvalue=None):
    '''Group elements of iterable in groups of n. For example:
       >>> [e for e in grouper(3, [1,2,3,4,5,6,7])]
       [(1, 2, 3), (4, 5, 6), (7, None, None)]'''
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def mt_map(threads, func, operands):
    '''Multithreaded map if threads > 1. threads = 1 is useful for debugging.'''
    if threads > 1:
        p = Pool(threads)
        return p.map(func, operands)
    else:
        return map(func, operands)


def linear(args, output_size, bias, bias_start=0.0, scope=None, initializer=None):
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
    if not tf.nn.nest.is_sequence(args):
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
