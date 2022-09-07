#!/usr/bin/env python3

from __future__ import print_function, division

import itertools as it


def powerset(iterable, start=0):
    """
    Set of all subsets:
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    `start` defines the minimum number of items to have in each output subset:
        start=0 will include the empty set;
        start=n will include all subsets of cardinality >= n

    Adapted from: https://docs.python.org/3/library/itertools.html
    """
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r)
                                  for r in range(start, len(s)+1))


def print_mis(all_mis, layer_sizes, header):
    print(header)
    for i, mis in enumerate(all_mis):
        print(i, end='', flush=True)
        for js in powerset(range(layer_sizes[i]), start=1):
            print('\t%.4f' % all_mis[i][js], end='', flush=True)
        print()


def print_edge_data(edge_data, layer_sizes):
    num_layers = len(layer_sizes)
    for j in range(max(layer_sizes[1:])):
        for i in range(0, num_layers - 1):
            if j >= layer_sizes[1:][i]:
                print(' ' * 7, end='\t')
            else:
                for k in range(layer_sizes[:-1][i]):
                    print('%7.4f' % edge_data[i][j, k].item(), end='\t')
            print('\t', end='')
        print()


def print_node_data(node_data, layer_sizes):
    num_layers = len(layer_sizes)
    for j in range(max(layer_sizes)):
        for i in range(num_layers):
            if j >= layer_sizes[i]:
                print(' ' * 7, end='\t')
            else:
                print('%7.4f' % node_data[i][j], end='\t')
            print('\t', end='')
        print()
