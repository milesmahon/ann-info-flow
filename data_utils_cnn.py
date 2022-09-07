#!/usr/bin/env python3

from __future__ import print_function, division

import joblib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as spl


def init_data(params, data=None):
    if data is None:
        data = SimpleNamespace()
    dataset = params.dataset
    data.dataset = dataset

    if dataset == 'mnist':
        d = np.load('mnist_train.npy',allow_pickle=True)
        params.num_train = d.shape[0]
        d = np.vstack((d,np.load('mnist_test.npy',allow_pickle=True)))
        params.num_data = d.shape[0]

        fives = np.where(d[:,1] == 5,1,0)
        labels = fives + np.where(d[:,1] == 8,1,0)     # 4,9=0; 5,8=1
        bias = np.where(d[:,1] % 2 == 0, 0, 1)         # Even=0; Odd=1
        X = d[:,0]
        Y = labels
        Z = bias

        # Choose a balanced sample of output classes
        class_inds = [np.where((Y == 0) & (Z == 0))[0],  # Even; 4,9
                      np.where((Y == 1) & (Z == 0))[0],  # Even; 5,8
                      np.where((Y == 0) & (Z == 1))[0],  # Odd; 4,9
                      np.where((Y == 1) & (Z == 1))[0]]  # Odd; 5,8

        # Set relative sizes of each class in the train and test sets
        train_weights = np.array([1, 2, 2, 1], dtype=float)
        test_weights = np.array([1, 2, 2, 1], dtype=float)
        weights = train_weights + test_weights
        class_sizes = np.array([class_ind.size for class_ind in class_inds])
        min_ind = np.argmin(class_sizes / weights)
        min_class_size = class_sizes[min_ind]
        norm_factor = weights[min_ind]
        weights /= norm_factor
        train_weights /= norm_factor
        test_weights /= norm_factor

        # Choose required number of train and test indices
        rng = np.random.default_rng(42)
        inds = [rng.choice(class_ind, size=int(weight * min_class_size),
                           replace=False, shuffle=False)
                for weight, class_ind in zip(weights, class_inds)]
        # Select required number of train indices from this set
        train_inds = [ind[:int(train_weight * min_class_size)]
                      for train_weight, ind in zip(train_weights, inds)]
        test_inds = [ind[int(train_weight * min_class_size):]
                     for train_weight, ind in zip(train_weights, inds)]
        # Concatenate, round off, and shuffle
        train_inds = np.concatenate(train_inds)
        train_inds_trunc = train_inds[:(train_inds.size // 10) * 10]
        test_inds = np.concatenate(test_inds)
        test_inds_trunc = test_inds[:(test_inds.size // 10) * 10]

        inds_trunc = np.concatenate((train_inds_trunc, test_inds_trunc))
        # Should NOT shuffle after this! We need to maintain the individual
        # statistics of the train and test sets

        X = X[inds_trunc]
        Y = Y[inds_trunc]
        Z = Z[inds_trunc]

        params.num_data = inds_trunc.size
        params.num_train = train_inds_trunc.size

        # Standardize features based on training data
        X = X - X[:params.num_train].mean().mean()
        X = X / X[:params.num_train].std().std()
        #X -= X[:params.num_train].mean(axis=0)
        #X /= X[:params.num_train].std(axis=0)

        data.data = (X, Y, Z)

    else:
        raise ValueError('Unknown dataset %s' % dataset)

    return data


def print_data_stats(data, params):
    X, Y, Z = data[:3]
    #X = np.array(X)
    #Y = np.array(Y)
    #if data.dataset == 'adult': # Adult dataset: 0 for race; 1 for gender
    #    Z = np.array(Z)[:, 1]
    #else:                       # Others (incl Tiny SCM): only one protected attr
    #Z = np.array(Z)

    class_inds = [np.where((Y[:params.num_train] == 0) & (Z[:params.num_train] == 0))[0],  # Men, <50K
                  np.where((Y[:params.num_train] == 1) & (Z[:params.num_train] == 0))[0],  # Men, >50K
                  np.where((Y[:params.num_train] == 0) & (Z[:params.num_train] == 1))[0],  # Women, <50K
                  np.where((Y[:params.num_train] == 1) & (Z[:params.num_train] == 1))[0]]  # Women, >50K
    print([ci.size for ci in class_inds])
    class_inds = [np.where((Y[params.num_train:] == 0) & (Z[params.num_train:] == 0))[0],  # Men, <50K
                  np.where((Y[params.num_train:] == 1) & (Z[params.num_train:] == 0))[0],  # Men, >50K
                  np.where((Y[params.num_train:] == 0) & (Z[params.num_train:] == 1))[0],  # Women, <50K
                  np.where((Y[params.num_train:] == 1) & (Z[params.num_train:] == 1))[0]]  # Women, >50K
    print([ci.size for ci in class_inds])


def compute_y(uy, ug, alpha=1, offset=0):
    w = []
    for i, (u, v) in enumerate(zip(uy + offset, ug + offset)):
        ##print(i, end=' ', flush=True)
        #def f(x):
        #    return (x[0] - u)**2 + (x[1] - v)**2
        #def c(x):
        #    return x[0] * x[1] - 1
        #res = opt.minimize(f, [1, 1], method='SLSQP',
        #                   constraints=dict(type='eq', fun=c))
        #dist = np.sqrt(res.fun)

        # If (x0, y0) is the foot of the perpendicular dropped from (u, v)
        # onto y = 1/x, then x0^4 - u.x0^2 + v.x0 - 1 = 0
        x0 = np.roots([1, 0, -u, v, -1])
        # Choose real roots with x0 > 0
        x0 = np.real(x0[np.isclose(np.imag(x0), 0)])
        y0 = 1 / x0
        dist = np.min(np.sqrt((x0 - u)**2 + (y0 - v)**2))

        # Add sign to the distance
        if u < 0 or v < 0:
            dist = -dist
        elif v < 1 / u:
            dist = -dist
        w.append(dist)
    w = np.array(w)
    y = (np.random.rand(uy.size) < spl.expit(alpha * w))
    #print(np.c_[uy, ug, w])
    return y.astype(int)


def compute_x_biased(u, z, alpha, beta, sigma):
    x = sigma * np.random.randn(*z.shape)
    x[z == 1] += alpha * u[z == 1]
    x[z == 0] += beta * u[z == 0]
    return x


def generate_data(n):
    np.random.seed(97)  # Was 7

    # Gaussian distribution
    uy = np.random.randn(n)
    ug = np.random.randn(n)
    # Scaled and centered uniform distribution
    #uy = np.random.rand(n) * 4 - 2
    #ug = np.random.rand(n) * 4 - 2
    u = np.array([uy, ug]).T

    z = (np.random.rand(n) < 0.5).astype(int)

    y = compute_y(uy, ug, 3, 1)
    #y = compute_y(uy, ug, 100, 1)
    x1 = compute_x_biased(uy, z, 0.7, 0.0, 0.2)
    x2 = compute_x_biased(uy, z, 0.5, 0.0, 0.2)
    x3 = compute_x_biased(ug, z, 0.1, 0.1, 0.2)  # No bias
    x = np.array([x1, x2, x3]).T

    # NOTE: There is some weird ordering effect coming from the above methods.
    # These arrays need to be permuted before use
    return x, y, z, u


if __name__ == '__main__':
    n = 1000
    x, y, z, u = generate_data(n)
    (x1, x2, x3) = x.T
    (uy, ug) = u.T

    y1 = (y == 1)
    y0 = ~y1
    z1 = (z == 1)
    z0 = ~z1

    mask_combinations = [z0 & y0, z0 & y1, z1 & y0, z1 & y1]
    params = [{'color': 'C1', 'mec': 'k'},
              {'color': 'C0', 'mec': 'k'},
              {'color': 'C1',},
              {'color': 'C0',},]

    #plt.figure()
    #for mask, p in zip(mask_combinations, params):
    #    plt.plot(ug[mask], uy[mask], 'o', alpha=0.3, **p)

    #plt.figure()
    #for mask, p in zip(mask_combinations, params):
    #    plt.plot(x2[mask], x3[mask], 'o', alpha=0.3, **p)

    plt.figure()
    plt.plot(np.linspace(1/4, 4, 100) - 1, 1 / np.linspace(1/4, 4, 100) - 1, 'k-')
    plt.plot(uy[y == 1], ug[y == 1], 'C0o', alpha=0.4)
    plt.plot(uy[y == 0], ug[y == 0], 'C1o', alpha=0.4)
    plt.xlabel('$U_Y$', fontsize=18)
    plt.ylabel('$U_G$', fontsize=18)
    plt.legend(('True boundary', '$Y=1$', '$Y=0$'), loc='lower left', fontsize=14)

    plt.figure()
    plt.plot(uy[z == 1], x1[z == 1], 'C0o', alpha=0.3)
    plt.plot(uy[z == 0], x1[z == 0], 'C1o', alpha=0.3)
    plt.xlabel('$U_Y$', fontsize=18)
    plt.ylabel('$X_1$', fontsize=18)
    plt.legend(('$Z=1$', '$Z=0$'), loc='best', fontsize=14)

    plt.figure()
    plt.plot(uy[z == 1], x2[z == 1], 'C0o', alpha=0.3)
    plt.plot(uy[z == 0], x2[z == 0], 'C1o', alpha=0.3)
    plt.xlabel('$U_Y$', fontsize=18)
    plt.ylabel('$X_2$', fontsize=18)
    plt.legend(('$Z=1$', '$Z=0$'), loc='best', fontsize=14)

    plt.figure()
    plt.plot(ug, x3, 'C0o', alpha=0.3)
    plt.xlabel('$U_G$', fontsize=18)
    plt.ylabel('$X_3$', fontsize=18)

    plt.show()
