#!/usr/bin/env python3

from __future__ import print_function, division

import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from param_utils import init_params


def fit_line(x, y):
    """
    Fit a line without an intercept and compute the R^2 value.
    `x` and `y` should 1D arrays.
    """

    # Estimated slope using pseudo-inverse
    mhat = (x @ y) / np.sum(x**2)
    # R^2 value: fraction of explained variance
    r2val = 1 - np.sum((y - mhat * x)**2) / np.sum((y - y.mean())**2)

    return mhat, r2val


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        params = init_params()
        dataset = params.dataset

    results_dir = 'results-%s' % dataset

    data = np.load(results_dir + '/scaling.npz')
    num_runs = data['acc_flows'].shape[0]
    acc_flows = data['acc_flows'].flatten()
    delta_accs = data['delta_accs'].flatten()
    bias_flows = data['bias_flows'].flatten()
    delta_biases = data['delta_biases'].flatten()

    title_kwargs = dict(fontsize=18)
    label_kwargs = dict(fontsize=16)

    # Plot acc/bias change against signed weighted information flow measures
    plt.figure()
    plt.plot(acc_flows, delta_accs, 'C0o')
    # TODO: Estimate a regression line through all these data points
    plt.title('$\Delta_{acc}$ vs signed weighted acc flow\n(%s dataset)' % dataset, **title_kwargs)
    plt.xlabel('Signed weighted accuracy flow', **label_kwargs)
    plt.ylabel('Change in output acc upon pruning\n(new acc - old acc)', **label_kwargs)
    plt.tight_layout()

    plt.figure()
    plt.plot(bias_flows, delta_biases, 'C1o')
    plt.title('$\Delta_{bias}$ vs signed weighted bias flow\n(%s dataset)' % dataset, **title_kwargs)
    plt.xlabel('Signed weighted bias flow', **label_kwargs)
    plt.ylabel('Change in output bias upon pruning\n(new bias - old bias)', **label_kwargs)
    plt.tight_layout()

    # Plot acc/bias change against absolute weighted information flows
    plt.figure()
    plt.plot(abs(acc_flows), delta_accs, 'C0o')
    # Compute regression lines for delta_acc & delta_bias vs. resp flow
    slope, intercept, rval = stats.linregress(abs(acc_flows), delta_accs)[:3]
    x = np.r_[min(abs(acc_flows)), max(abs(acc_flows))]
    plt.plot(x, slope * x + intercept, 'k-')
    #slope, r2val = fit_line(abs(acc_flows), delta_accs)
    #x = np.r_[min(abs(acc_flows)), max(abs(acc_flows))]
    #plt.plot(x, slope * x, 'k-')
    #print(r2val)
    plt.title('$\Delta_{acc}$ vs abs weighted acc flow ($R^2 = %.2f$)\n(%s dataset)' % (rval**2, dataset), **title_kwargs)
    plt.xlabel('Absolute weighted accuracy flow', **label_kwargs)
    plt.ylabel('Change in output acc upon pruning\n(new acc - old acc)', **label_kwargs)
    plt.tight_layout()

    plt.figure()
    plt.plot(abs(bias_flows), delta_biases, 'C1o')
    # Compute regression lines for delta_acc & delta_bias vs. resp flow
    slope, intercept, rval = stats.linregress(abs(bias_flows), delta_biases)[:3]
    x = np.r_[min(abs(bias_flows)), max(abs(bias_flows))]
    plt.plot(x, slope * x + intercept, 'k-')
    #slope, r2val = fit_line(abs(bias_flows), delta_biases)
    #x = np.r_[min(abs(bias_flows)), max(abs(bias_flows))]
    #plt.plot(x, slope * x, 'k-')
    #print(r2val)
    plt.title('$\Delta_{bias}$ vs abs weighted bias flow ($R^2 = %.2f$)\n(%s dataset)' % (rval**2, dataset), **title_kwargs)
    plt.xlabel('Absolute weighted bias flow', **label_kwargs)
    plt.ylabel('Change in output bias upon pruning\n(new bias - old bias)', **label_kwargs)
    plt.tight_layout()

    # Plot acc/bias change against acc/bias ratio
    plt.figure()
    plt.plot(abs(acc_flows)/abs(bias_flows), delta_accs, 'C1o')
    plt.title('$\Delta_{acc}$ vs acc/bias flow ratio\n(%s dataset)' % dataset, **title_kwargs)
    plt.xlabel('Acc/bias flow ratio', **label_kwargs)
    plt.ylabel('Change in output acc upon pruning\n(new acc - old acc)', **label_kwargs)
    plt.tight_layout()

    plt.figure()
    plt.plot(abs(bias_flows)/abs(acc_flows), delta_biases, 'C1o')
    plt.title('$\Delta_{bias}$ vs bias/acc flow ratio\n(%s dataset)' % dataset, **title_kwargs)
    plt.xlabel('Bias/Acc flow ratio', **label_kwargs)
    plt.ylabel('Change in output bias upon pruning\n(new bias - old bias)', **label_kwargs)
    plt.tight_layout()

    # TODO: Plot weighted acc/bias flow ratios - to do this, it would be easier
    # to just load from the analyzed-data file directly.

    plt.show()
