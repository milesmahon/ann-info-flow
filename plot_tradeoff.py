#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = 'adult'
    tradeoff_files = [
        'results/tradeoff-biasacc-node-1.npz',
        'results/tradeoff-biasacc-node-2.npz',
        'results/tradeoff-accbias-node-1.npz',
        'results/tradeoff-accbias-node-2.npz',
        'results/tradeoff-biasacc-edge-1.npz',
        'results/tradeoff-biasacc-edge-2.npz',
        'results/tradeoff-accbias-edge-1.npz',
        'results/tradeoff-accbias-edge-2.npz',
    ]
    colors = ['C0-o', 'C1-o', 'C0-s', 'C1-s', 'C2--o', 'C3--o', 'C2--s', 'C3--s']
    legend = [
        'biasacc-node-1',
        'biasacc-node-2',
        'accbias-node-1',
        'accbias-node-2',
        'biasacc-edge-1',
        'biasacc-edge-2',
        'accbias-edge-1',
        'accbias-edge-2',
    ]

    plt.figure()

    for tradeoff_file, color in zip(tradeoff_files, colors):
        data = np.load(tradeoff_file)
        accs = data['accs']
        biases = data['biases']
        plt.plot(biases, accs, color)

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
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(legend, title='Configuration', loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(legend, loc='best', title='Configuration', fontsize=14)
    plt.savefig('figures/bias-acc-tradeoff.png')

    plt.show()
