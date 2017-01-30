from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def partial_tokenize(args):
    patients_list, (shlf_file, note_type) = args
    if note_type:
        note_type = note_type.replace('_', ' ')
    shelf = shelve.open(shlf_file)
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
            adm_map[adm.admission_id] = []
            for note in adm.nte_events:
                if not note_type or note.note_cat == note_type:
                    found = True
                    note_text = []
                    for sent in mimic_tokenize(note.note_text):
                        note_text.append(sent)
                    adm_map[adm.admission_id].append(note_text)
        if found:
            ret[pid] = adm_map
    shelf.close()
    return ret


def partial_read(args):
    patients_list, (pshlf_file, nshlf_file) = args
    pshelf = shelve.open(pshlf_file)
    nshelf = shelve.open(nshlf_file)
    ret = []
    for pid in patients_list:
        patient_notes = nshelf[pid]
        for adm_id, note in patient_notes.items():
            ret.append((pshelf[pid].admissions[adm_id], note))
    nshelf.close()
    pshelf.close()
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
