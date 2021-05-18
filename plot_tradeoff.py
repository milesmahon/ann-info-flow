#!/usr/bin/env python3

from __future__ import print_function, division

import sys
import numpy as np
import matplotlib.pyplot as plt
from param_utils import init_params


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        params = init_params()
        dataset = params.dataset

    results_dir = 'results-%s' % dataset
    with open(results_dir + '/combos.txt') as f:
        combos = [line.strip() for line in f.readlines()]

    tradeoff_files = ['tradeoff-%s.npz' % combo for combo in combos]
    colors = ['C0-o', 'C1-o', 'C0--s', 'C1--s', 'C2-o', 'C3-o', 'C2--s', 'C3--s']
    #colors = ['C0-o', 'C1-o', 'C2-o', 'C3-o']
    #colors = ['C3-o']
    legend = combos

    plt.figure()

    avg_lines = []
    for tradeoff_file, color in zip(tradeoff_files, colors):
        data = np.load(results_dir + '/' + tradeoff_file)
        accs = data['accs']
        biases = data['biases']
        #plt.plot(biases.T, accs.T, color[:-1], alpha=0.5, linewidth=1)
        avg_line, = plt.plot(biases.mean(axis=0), accs.mean(axis=0), color)
        avg_lines.append(avg_line)

    #plt.axis('square')
    ax = plt.gca()
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()
    #newlim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    #ax.set_xlim(newlim)
    #ax.set_ylim(newlim)

    #plt.plot([0, 1], [0, 1], 'k-', linewidth=1, zorder=-1)
    plt.title('Bias-accuracy tradeoffs\n(%s dataset)' % dataset, fontsize=18)
    plt.xlabel('Bias', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    # Put legend outside plot
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    #ax.legend(avg_lines, legend, title='Configuration', loc='center left',
    #          bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=14)
    ax.legend(avg_lines, legend, loc='upper left', title='Configuration', fontsize=12,
              title_fontsize=14)
    plt.axis('equal')
    plt.grid(color=[0.9, 0.9, 0.9])
    plt.savefig(results_dir + '/bias-acc-tradeoff.png')

    plt.show()
