#!/usr/bin/env python3

from __future__ import print_function, division

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from param_utils import init_params


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        params = init_params()
        dataset = params.dataset

    if len(sys.argv) > 2:
        subfolder = sys.argv[2]
    else:
        subfolder = ''

    # subfolder = 'linear-svm'
    results_dir = 'results-%s' % dataset
    results_subfolder = os.path.join(results_dir, subfolder)

    with open(results_dir + '/combos.txt') as f:
        combos = [line.strip() for line in f.readlines()]

    tradeoff_files = ['tradeoff-%s.npz' % combo for combo in combos]
    #colors = ['C0-o', 'C1-o', 'C0--s', 'C1--s', 'C2-o', 'C3-o', 'C2--s', 'C3--s']
    #colors = ['C0-o', 'C1-o', 'C2-o', 'C3-o']
    #colors = ['C3-o']

    #colors = [cm.tab20c(i) for i in [0, 2, 4, 6, 8, 10]]
    colors = [cm.Paired(i) for i in [0, 1, 6, 7, 2, 3]]
    colors *= 2
    markers = ['o'] * 6 + ['s'] * 6
    linestyles = ['-'] * 6 + ['--'] * 6

    legend = []
    for combo in combos:
        metric, method, num_prune = combo.split('-')
        if metric == 'biasacc':
            metric = 'Bias/Acc'
        elif metric == 'accbias':
            metric = 'Acc/Bias'
        if method == 'node':
            method = 'node(s)'
        elif method == 'edge':
            method = 'edge(s)'
        elif method == 'path':
            method = 'path(s)'
        legend.append('%s: %s %s' % (metric, num_prune, method))

    info_method = 'Correlation'
    if subfolder == 'linear-svm':
        info_method = 'Linear SVM'
    elif subfolder == 'kernel-svm':
        info_method = 'Kernel SVM'

    plt.figure(figsize=(9, 6))

    avg_lines = []
    for tradeoff_file, color, marker, linestyle in zip(tradeoff_files, colors, markers, linestyles):
        data = np.load(results_subfolder + '/' + tradeoff_file)
        #accs = data['accs']
        #biases = data['biases']
        ##plt.plot(biases.T, accs.T, color[:-1], alpha=0.5, linewidth=1)
        #avg_line, = plt.plot(biases.mean(axis=0), accs.mean(axis=0), color)
        #avg_lines.append(avg_line)

        accs = data['accs'] * 100      # Convert to percentage
        biases = data['biases'] * 100
        num_runs = accs.shape[0]
        #avg_line = plt.plot(biases[0].T, accs[0].T, color[:-1], alpha=0.5, linewidth=1)
        #accs = accs - accs[:, -1].reshape(100, 1)
        #biases = biases - biases[:, -1].reshape(100, 1)
        acc_err = np.std(accs, axis=0) / np.sqrt(num_runs)
        bias_err = np.std(biases, axis=0) / np.sqrt(num_runs)
        #plt.errorbar(biases.mean(axis=0), accs.mean(axis=0), xerr=acc_err, yerr=bias_err, ls='none', ecolor=color.split('-')[0], capsize=2)
        plt.errorbar(biases.mean(axis=0), accs.mean(axis=0), xerr=acc_err, yerr=bias_err, ls='none', ecolor=color, capsize=2)
        avg_line, = plt.plot(biases.mean(axis=0), accs.mean(axis=0), color=color, marker=marker, linestyle=linestyle)
        #avg_line, = plt.plot(biases, accs, 'ro')
        avg_lines.append(avg_line)
    plt.plot(biases.mean(axis=0)[-1], accs.mean(axis=0)[-1], 'k*', markersize=15)

    #plt.axis('square')
    ax = plt.gca()
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()
    #newlim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    #ax.set_xlim(newlim)
    #ax.set_ylim(newlim)

    #plt.plot([0, 1], [0, 1], 'k-', linewidth=1, zorder=-1)
    if 'adult' in dataset:
        dataset = 'Adult'
    elif dataset == 'tinyscm':
        dataset = 'Synthetic'
    plt.title('Bias-accuracy tradeoff in a larger ANN (5x3x3x2)\n(Dataset: %s, MI est: %s)' % (dataset, info_method), fontsize=18)
    plt.xlabel('Bias (%)', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    # Put legend outside plot
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    #ax.legend(avg_lines, legend, title='Configuration', loc='center left',
    #          bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.grid(color=[0.9, 0.9, 0.9])
    #plt.axis('square')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    ax.legend(avg_lines, legend, title='Configuration', fontsize=16,
              title_fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(results_subfolder + '/bias-acc-tradeoff.png')

    plt.show()
