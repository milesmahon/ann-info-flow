#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import stats

from sklearn import preprocessing, svm, linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     cross_validate, RandomizedSearchCV)
from sklearn.kernel_approximation import Nystroem, RBFSampler

#import thundersvm

from utils import powerset


def Hb(p):
    """Binary entropy function."""
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)


def Hb_inv(y):
    """
    Inverse binary entropy function. Provides a result in the interval [0, 0.5].
    """
    if y == 0:
        return 0
    elif y == 1:
        return 0.5
    elif y > 1 or y < 0:
        raise ValueError('y should be between 0 and 1')

    p_approx = 0.5 * (1 - np.sqrt(1 - y))

    fn = lambda z: (y - Hb(z))
    jac = lambda z: np.log2(z / (1 - z))
    res = opt.root_scalar(fn, bracket=(0+1e-6, 0.5-1e-6), fprime=jac, x0=p_approx)

    if not res.converged:
        # TODO Make this a warning
        print('Hb_inv did not converge')
    return res.root


def acc_from_mi(mi, Hx=1):
    """
    Compute effective classification accuracy from mutual information.

    I(X ; Y) = H(X) - H(X | Y)
    H(X | Y) = H(X) - I(X ; Y)  => Randomness in X given Y
                                => Hb(Prob of error in inferring X given Y)
    Acc(X | Y) = 1 - Hb_inv(H(X) - I(X ; Y))  => Hb_inv(1 - Prob of error)
    """

    return 1 - Hb_inv(Hx - mi)


def mutual_info_bin(x, y, Hx=None, num_train=None, return_acc=False):
    """
    Compute mutual information between x and y, where x is assumed to be a
    binary (0/1) random variable.
    """

    # XXX: Hx estimation is not correct - Hx is computed on an x that has not
    # been re-scaled, whereas Hx_y is computed on a rescaled x - differential
    # entropy is not immune to scaling - same scaling should be applied to both

    # I don't know what the above complaint is referring to... X is not a
    # continuous random variable, so H(X) is not differential entropy!

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
    #classifier = svm.SVC(kernel='rbf')
    #classifier = thundersvm.SVC(kernel='rbf')
    feature_map = Nystroem(n_components=100)
    #feature_map = RBFSampler(n_components=100)
    #classifier = svm.LinearSVC(dual=False)
    classifier = linear_model.SGDClassifier(warm_start=True)
    # For LinearSVC, always set dual=False if number of features is less than
    # number of data points

    # Hyperparameters
    #Cs = np.logspace(-2, 2, 25)
    # TODO: Use randomized grid search CV
    #Cs = np.logspace(-2, 2, 5)
    #gammas = np.logspace(-2, 2, 5)
    C_dist = stats.loguniform(10**-2, 10**2)
    gamma_dist = stats.loguniform(10**-2, 10**2)

    num_train = x.size

    # Jointly train and test to estimate best classification accuracy
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=73)
    inner_cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=71)

    #pipe = Pipeline(steps=[('scaler', scaler), ('svc', classifier)])
    #estimator = GridSearchCV(pipe, dict(svc__C=Cs, svc__gamma=gammas),
    #                         cv=inner_cv)
    pipe = Pipeline(steps=[('scaler', scaler), ('fm', feature_map), ('svc', classifier)])
    #estimator = GridSearchCV(pipe, dict(fm__gamma=gammas, svc__C=Cs,),
    #                         cv=inner_cv)
    #estimator = GridSearchCV(pipe, dict(fm__gamma=gammas, svc__alpha=Cs,),
    #                         cv=inner_cv)
    estimator = RandomizedSearchCV(pipe, dict(fm__gamma=gamma_dist, svc__alpha=C_dist),
                                   cv=inner_cv, n_iter=25, random_state=53)

    #import warnings
    #with warnings.catch_warnings():
    #    warnings.filterwarnings("ignore", message="Liblinear failed to converge")
    cv_ret = cross_validate(estimator, y, x, cv=outer_cv, verbose=0,
                            return_train_score=False)

    acc = cv_ret['test_score'].mean()

    #cv_rets = []
    #for train, test in outer_cv.split(y, x):
    #    classifier = linear_model.SGDClassifier(warm_start=True)
    #    pipe = Pipeline(steps=[('scaler', scaler), ('fm', feature_map), ('svc', classifier)])
    #    estimator = GridSearchCV(pipe, dict(fm__gamma=gammas, svc__alpha=Cs,),
    #                             cv=inner_cv)
    #    estimator.fit(y[train], x[train])
    #    cv_rets.append(estimator.score(y[test], x[test]))
    #acc = np.array(cv_rets).mean()

    # Compute conditional entropy of x given y, and mutual information
    Hx_y = Hb(acc)
    Ixy = max(Hx - Hx_y, 0)

    # Return mutual information
    if return_acc:
        return Ixy, acc
    return Ixy


def cond_mi(mis, Xi, Xj):
    """
    Compute the conditional mutual information of X_i given X_j.

    Xi and Xj should be iterables.
    """
    Xi_U_Xj = tuple(sorted(set(Xi) | set(Xj)))
    return max(mis[Xi_U_Xj] - mis[Xj], 0)


def info_flow(mis, Xi, n):
    """
    Compute information flow for node i, given all mutual informations for a
    particular layer. n is the number of nodes in that layer.
    """
    flows = []
    for s in powerset(range(n)):
        if Xi in s:
            Xj = tuple(sorted(set(s) - {Xi,}))
            flows.append(cond_mi(mis, [Xi,], Xj))

    return max(flows)


def compute_all_flows(all_mis, layer_sizes):
    """
    Compute all information flows from all mutual information values.
    """

    all_flows = [np.zeros(layer_size) for layer_size in layer_sizes]
    for i, layer_size in enumerate(layer_sizes):
        for j in range(layer_size):
            all_flows[i][j] = info_flow(all_mis[i], j, layer_size)

    return all_flows


def weight_info_flows(all_flows, weights):
    """
    For each layer, multiply info flows measured for each node with the weights
    of the outgoing edges.

    Weighted flows are returned in the same format as the weights of the
    neural network, where weighted_flows[k][j, i] is the weighted flow going
    from node i in layer k to node j in layer k+1.
    """
    weighted_flows = [flows * w for flows, w in zip(all_flows[:-1], weights)]
    # Very last layer of nodes has no outgoing edges, so all weights are 1
    weighted_flows.append(all_flows[-1])

    return weighted_flows


if __name__ == '__main__':
    import scipy.linalg as la


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
