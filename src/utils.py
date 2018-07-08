from __future__ import print_function

import errno
import os

import numpy as np
import re
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from relation import Relation


def sliding_window(iterable, left, right, padding=None, step=1):
    """Make a sliding window iterator with padding.

    Iterate over `iterable` with a step size `step` producing a tuple for each element:
        ( ... left items, item, right_items ... )
    such that item visits all elements of `iterable` `step`-steps aside, the length of
    the left_items and right_items is `left` and `right` respectively, and any missing
    elements at the start and the end of the iteration are padded with the `padding`
    item.

    For example:

    >>> list( sliding_window( range(5), 1, 2 ) )
    [(None, 0, 1, 2), (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, None), (3, 4, None, None)]

    """
    from itertools import islice, repeat, chain
    from collections import deque

    n = left + right + 1

    iterator = chain(iterable, repeat(padding, right))

    elements = deque(repeat(padding, left), n)
    elements.extend(islice(iterator, right - step + 1))

    while True:
        for i in range(step):
            elements.append(next(iterator))
        yield tuple(elements)


def print_stats(y_true, y_pred, labels_index, binary, map_argmax = True):
    if not binary and map_argmax:
        y_true = map(lambda v: v.argmax(), y_true)
        y_pred = map(lambda v: v.argmax(), y_pred)



    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)

    acc = accuracy_score(y_true, y_pred)
    print("acc: ", acc)

    p_r_f = precision_recall_fscore_support(y_true, y_pred)
    print_p_r_f(p_r_f, labels_index)

    lab = range(1, len(labels_index))

    p_r_f_avg_micro = precision_recall_fscore_support(y_true, y_pred, average='micro', labels=lab)
    print ('micro averaged:')
    print(p_r_f_avg_micro)

    print ('weighted averaged:')
    p_r_f_avg_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=lab)
    print(p_r_f_avg_weighted)

    return p_r_f, acc, p_r_f_avg_micro, p_r_f_avg_weighted


def print_p_r_f(p_r_f, labels_index):
    print("")
    print("  \t", end='')

    for key, value in sorted(labels_index.iteritems(), key=lambda x: x[1]):
        print(key + "\t\t", end='')
    print("")
    print("P \t", end='')
    for p in p_r_f[0]:
        print(str(p) + "\t", end='')
    print("")
    print("R \t", end='')
    for p in p_r_f[1]:
        print(str(p) + "\t", end='')
    print("")
    print("F1\t", end='')
    for p in p_r_f[2]:
        print(str(p) + "\t", end='')

    print("")


def split(y_data, split_ratio, balanced=True):
    if balanced:
        return balanced_split(y_data, split_ratio)

    return simple_split(y_data, split_ratio)


def simple_split(y_data, split_ratio):

    y1_target_len = split_ratio * len(y_data)
    y1 = []
    y2 = []
    for i in range(len(y_data)):
        if i < y1_target_len:
            y1.append(i)
        else:
            y2.append(i)
    return y1, y2


def balanced_split(y_data, split_ratio):
    y_true = map(lambda v: v.argmax(), y_data)

    class_num = len(np.unique(y_true))
    print(class_num)
    class_num_arr = np.zeros(class_num)
    for y in y_true:
        class_num_arr[y] += 1

    y1_class_num_arr = class_num_arr * split_ratio
    y1_class_num_arr_curr = np.zeros(class_num)
    y1 = []
    y2 = []

    for i, y_d in enumerate(y_data):
        y = y_d.argmax()

        if y1_class_num_arr_curr[y] < y1_class_num_arr[y]:
            y1.append(i)
            y1_class_num_arr_curr[y] +=1
        else:
            y2.append(i)

    print(y1_class_num_arr, y1_class_num_arr_curr)
    return y1, y2


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def merge_several_folds_results(data):
    data = [[d if d is not None else 0 for d in row] for row in data]
    a = np.array(data[0])
    for i in range(1, len(data)):
        if not (a is None or data[i] is None):
            a += np.array(data[i])
    a /= len(data)
    return a

def get_annotation_position(a):
    return (a.getSentence().getOrd(), a.getHead())


def get_relation_position(r):
    return (get_annotation_position(r.getAnnotationFrom()), get_annotation_position(r.getAnnotationTo()))

def create_relation_dictionary(config, document, rel_dict=None):
    if not rel_dict:
        rel_dict = {}

    for r in document.getRelations().getRelations():
        rel_type = r.getType()

        matching_rel_configs = [rc for rc in config['relations'] if rel_type in rc["types"]]
        if not len(matching_rel_configs):
            continue
        annotation_from = r.getAnnotationFrom()
        annotation_to = r.getAnnotationTo()

        for rel_conf in matching_rel_configs:

            if (any(re.match(_from, annotation_from.getType()) for _from in rel_conf['from']) and any(re.match(_to, annotation_to.getType()) for _to in rel_conf['to'])) \
                    or ("allow_reversed" in rel_conf and rel_conf["allow_reversed"] and any(re.match(_from, annotation_from.getType()) for _from in rel_conf['to']) and any(
                        re.match(_to, annotation_to.getType()) for _to in rel_conf['from'])):
                relation_position = get_relation_position(r)
                # print(rel_type, annotation_from.getType(), annotation_to.getType())
                # if relation_position in rel_dict:
                #     print(relation_position, "already in dict", rel_type, rel_dict[relation_position].getType())

                rel_dict[relation_position] = Relation(rel_type, annotation_from, annotation_to)
                continue


                # if any(
                #         any(re.match(_from, annotation_from.getType()) for _from in rel_conf['from']) and
                #         any(re.match(_to, annotation_to.getType()) for _to in rel_conf['to'])
                #         for rel_conf in matching_rel_configs):
                #     relation_position = self.get_relation_position(r)
                #     true_rel_dict[relation_position] = Relation(rel_type, annotation_from, annotation_to)

    return rel_dict

