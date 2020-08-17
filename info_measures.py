#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     cross_validate)


def Hb(p):
    """Binary entropy function."""
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)


def mutual_info_bin(x, y, Hx=None, num_train=None, return_acc=False):
    """
    Compute mutual information between x and y, where x is assumed to be a
    binary (0/1) random variable.
    """

    # XXX: Hx estimation is not correct - Hx is computed on an x that has not
    # been re-scaled, whereas Hx_y is computed on a rescaled x - differential
    # entropy is not immune to scaling - same scaling should be applied to both

    # Use provided entropy of x, or compute from data
    if (type(Hx) is float or type(Hx) is int) and (0 <= Hx <= 1):
        pass
    elif Hx is None:
        p_hat = x.mean()
        Hx = Hb(p_hat)
    else:
        raise ValueError('Hx should be a number between 0 and 1, or left as '
                         'None to be estimated from data')

    # x and y are exchanged from their usual positions here: x represents the
    # "labels" and y represents the "features" in the classification problem
    x = np.array(x)
    y = np.array(y)

    num_total = x.size
    if num_train is None:
        num_train = int(0.75 * num_total)

    # Classifier objects
    scaler = preprocessing.StandardScaler()
    classifer = svm.SVC(kernel='rbf')

    # Hyperparameters
    Cs = np.logspace(-2, 2, 5)
    gammas = np.logspace(-2, 2, 5)
    num_train = x.size

    # Jointly train and test to estimate best classification accuracy
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
    inner_cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)

    pipe = Pipeline(steps=[('scaler', scaler), ('svc', classifer)])
    estimator = GridSearchCV(pipe, dict(svc__C=Cs, svc__gamma=gammas),
                             cv=inner_cv, iid=False)
    # iid=False is default, but it throws a depracation warning unless
    # explicitly passed to the function

    cv_ret = cross_validate(estimator, y, x, cv=outer_cv, verbose=0,
                            return_train_score=False)
    acc = cv_ret['test_score'].mean()

    # Compute conditional entropy of x given y, and mutual information
    Hx_y = Hb(acc)
    Ixy = max(Hx - Hx_y, 0)

    # Return mutual information
    if return_acc:
        return Ixy, acc
    return Ixy


if __name__ == '__main__':
    import scipy.linalg as la
    from scipy import stats


    def gen_random_orthogonal_matrix(d):
        A = np.random.randn(d, d)
        Q, R = la.qr(A)
        lamda = R.diagonal().copy()
        lamda /= np.abs(lamda)
        return np.dot(Q, np.diag(lamda))


    def gen_data(n, d, c, p=0.5):
        """
        Generate a two-class dataset for classification.

        n: number of data points
        d: dimensionality of each data point
        c: max. classification accuracy (distinguishability)
        p: probability of the first class
        """

        y = np.random.rand(n)
        y = np.where(y <= p, 0, 1)

        mu = stats.norm.ppf(c)
        t = np.empty((d, n))
        t[0, :] = mu * (1 - 2*y) + np.random.randn(n)
        if d > 1:
            t[1:, :] = np.random.randn(d-1, n)

        Q = gen_random_orthogonal_matrix(d)
        X = np.dot(Q, t)

        return (X, y, Q)  # (features, labels, transform)


    num_total = 200
    num_dims = 2

    max_accs = np.array([0.51, 0.6, 0.7, 0.8, 0.9, 0.99])
    mis = []
    for max_acc in max_accs:
        print('%.2g: ' % max_acc, end='', flush=True)

        (y, x, Q) = gen_data(num_total, num_dims, max_acc)
        actual_mi = 1 - Hb(max_acc)
        est_mi, acc = mutual_info_bin(x, y.T, Hx=1, return_acc=True)
        mis.append(est_mi)

        print(acc, est_mi)

    plt.plot(max_accs, 1 - Hb(max_accs))
    plt.plot(max_accs, mis)
    plt.show()
