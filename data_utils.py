#!/usr/bin/env python3

from __future__ import print_function, division

import joblib
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as spl


def init_data(params, dataset=None, data=None):
    if data is None:
        data = SimpleNamespace()
    if dataset is None:
        dataset = params.default_dataset

    if dataset == 'TinySCM':
        # Load data if it exists; generate and save data otherwise
        if params.datafile is None:
            data.data = generate_data(params.num_data)
            print('Data generation complete')
            #joblib.dump(data, datafile, compress=3)
        else:
            data.data = joblib.load(params.datafile)

    else:
        raise ValueError('Unknown dataset %s' % dataset)

    return data


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
    # Gaussian distribution
    uy = np.random.randn(n)
    ug = np.random.randn(n)
    # Scaled and centered uniform distribution
    #uy = np.random.rand(n) * 4 - 2
    #ug = np.random.rand(n) * 4 - 2

    z = (np.random.rand(n) < 0.5).astype(int)

    y = compute_y(uy, ug, 3, 1)
    #y = compute_y(uy, ug, 100, 1)
    x1 = compute_x_biased(uy, z, 0.7, 0.0, 0.2)
    x2 = compute_x_biased(uy, z, 0.5, 0.0, 0.2)
    x3 = compute_x_biased(ug, z, 0.1, 0.1, 0.2)  # No bias

    # Return z, u, x, y
    return z, (uy, ug), (x1, x2, x3), y


if __name__ == '__main__':
    n = 500
    z, (uy, ug), (x1, x2, x3), y = generate_data(n)

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
    plt.plot(uy[z == 1], x1[z == 1], 'C0o', alpha=0.3)
    plt.plot(uy[z == 0], x1[z == 0], 'C1o', alpha=0.3)
    plt.xlabel('$U_Y$', fontsize=18)
    plt.ylabel('$X_1$', fontsize=18)

    plt.figure()
    plt.plot(uy[z == 1], x2[z == 1], 'C0o', alpha=0.3)
    plt.plot(uy[z == 0], x2[z == 0], 'C1o', alpha=0.3)
    plt.xlabel('$U_Y$', fontsize=18)
    plt.ylabel('$X_2$', fontsize=18)

    plt.figure()
    plt.plot(ug, x3, 'C0o', alpha=0.3)
    plt.xlabel('$U_G$', fontsize=18)
    plt.ylabel('$X_3$', fontsize=18)

    plt.show()
