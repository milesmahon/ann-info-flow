#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import joblib
from param_utils import init_params


def init_plots(vis):
    # Initialize the plotting axes and add them to the namespace
    vis.fig = plt.figure(figsize=(15, 5))

    vis.fig.canvas.toolbar_visible = False
    vis.fig.canvas.header_visible = False
    vis.fig.canvas.footer_visible = False
    vis.fig.canvas.resizable = False

    vis.ax_weights = vis.fig.add_subplot(1, 3, 1, aspect='equal')
    vis.ax_weights.set_axis_off()
    vis.ax_weights.set_title('ANN Weights')

    vis.ax_bias_flows = vis.fig.add_subplot(1, 3, 2, aspect='equal')
    vis.ax_bias_flows.set_axis_off()
    vis.ax_bias_flows.set_title('Weighted Bias Flows')

    vis.ax_acc_flows = vis.fig.add_subplot(1, 3, 3, aspect='equal')
    vis.ax_acc_flows.set_axis_off()
    vis.ax_acc_flows.set_title('Weighted Accuracy Flows')

    plt.tight_layout()


def plot_ann(layer_sizes, weights, plot_params=None, ax=None, flow_type='bias', info_method=''):
    """
    Plots a visualization of a trained neural network with given weights.

    Based on a gist by Colin Raffel
    (https://gist.github.com/craffel/2d727968c3aaebd10359)
    """

    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        plt.axis('off')
    if plot_params is None:
        plot_params = (0.1, 0.9, 0.9, 0.1)
    left, right, top, bottom = plot_params

    num_layers = len(layer_sizes)
    v_spacing = (top - bottom) / max(layer_sizes)
    h_spacing = (right - left) / (num_layers - 1)

    # Plot circles for neurons
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2 + (top + bottom) / 2
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing),
                                h_spacing/4, color='w', ec='k', zorder=4)
            ax.add_artist(circle)

    # Normalize edge weights to lie between -1 and +1
    max_weight = max(np.abs(w).max() for w in weights)
    normalized_weights = [w / max_weight for w in weights]
    alpha0 = 0.3
    t0, tmax = (0.5, 5)

    # Plot lines for edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2 + (top + bottom)/2
        layer_top_b = v_spacing*(layer_size_b - 1)/2 + (top + bottom)/2
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                weight = normalized_weights[n].reshape((layer_size_b, layer_size_a))[o, m]
                color = 'C0' if flow_type == 'acc' else 'C1'
                alpha = alpha0 + abs(weight) * (1 - alpha0)
                thickness = t0 + abs(weight) * (tmax - t0)

                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],
                                  color=color, alpha=alpha, linewidth=thickness)
                ax.add_artist(line)

    if flow_type == 'bias':
        ax.set_title("Bias flow visualization\n(Dataset: Synthetic, MI est: %s)" % info_method ,fontsize=18)
    else:
        ax.set_title("Accuracy flow visualization\n(Dataset: Synthetic, MI est: %s)" % info_method ,fontsize=18)

    return ax


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print("Please provide the dataset (adult, tinyscm), MI estimation method (corr,linear-svm,kernel-svm), and run number.")
        exit()

    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        params = init_params()
        dataset = params.dataset

    if len(sys.argv) > 2:
        subfolder = sys.argv[2]
    else:
        subfolder = ''

    if len(sys.argv) > 3:
        run = int(sys.argv[3])
    else:
        run = 0

    #subfolder = 'linear-svm'
    results_dir = 'results-%s' % dataset
    results_subfolder = os.path.join(results_dir, subfolder)

    params=init_params(dataset=dataset)
    nets=joblib.load(params.annfile)
    layer_sizes = nets[run].layer_sizes

    #weights = nets[0].get_weights() # Plot just the network weights.
    bias_flows=joblib.load(results_subfolder + '/analyzed-data.pkl')[run][2][:-1]
    weights = [abs(w) for w in bias_flows]
    plot_ann(layer_sizes, weights, flow_type='bias', info_method=subfolder)

    #weights = nets[0].get_weights()
    acc_flows=joblib.load(results_subfolder + '/analyzed-data.pkl')[run][5][:-1]
    weights = [abs(w) for w in acc_flows]
    plot_ann(layer_sizes, weights, flow_type='acc', info_method=subfolder)
    plt.show()
