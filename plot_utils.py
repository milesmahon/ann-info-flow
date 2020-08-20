#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt


def plot_ann(layer_sizes, weights, params=None, ax=None):
    """
    Plots a visualization of a trained neural network with given weights.

    Based on a gist by Colin Raffel
    (https://gist.github.com/craffel/2d727968c3aaebd10359)
    """

    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        plt.axis('off')
    if params is None:
        params = (0.1, 0.9, 0.9, 0.1)
    left, right, top, bottom = params

    num_layers = len(layer_sizes)
    v_spacing = (top - bottom) / max(layer_sizes)
    h_spacing = (right - left) / (num_layers - 1)

    # Plot circles for neurons
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2 + (top + bottom) / 2
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing),
                                v_spacing/4, color='w', ec='k', zorder=4)
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
                color = 'C0' if weight > 0 else 'C1'
                alpha = alpha0 + abs(weight) * (1 - alpha0)
                thickness = t0 + abs(weight) * (tmax - t0)

                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],
                                  color=color, alpha=alpha, linewidth=thickness)
                ax.add_artist(line)

    return ax


if __name__ == '__main__':
    layer_sizes = [3, 3, 1]
    weights = [
        np.array([[-0.6317, -0.7838, -2.1873],
                  [ 0.6223,  0.8153,  0.8304],
                  [-0.7554, -1.1385, -1.7987]]),
        np.array([-2.2715, 1.4479, -2.2133])
    ]

    plot_ann(layer_sizes, weights)
    plt.show()
