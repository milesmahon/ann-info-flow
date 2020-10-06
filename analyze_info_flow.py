#!/usr/bin/env python3

from __future__ import print_function, division

import os
import copy
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch

from data_utils import generate_data
from info_measures import (mutual_info_bin, compute_all_flows,
                           weight_info_flows, acc_from_mi)
from utils import powerset, print_mis, print_edge_data, print_node_data
from nn import SimpleNet, train_ann
from plot_utils import plot_ann


def corrcoef(x, y):
    """
    Correlate an (n, m) array and an (n, r) array to get an
    (m, r) cross-correlation matrix.
    """
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)

    cov = np.einsum('nm,nr->mr', x, y) / x.shape[0]
    sigma_x = np.std(x, axis=0)
    sigma_y = np.std(y, axis=0)

    return cov / (sigma_x[:, None] * sigma_y)


def analyze_info_flow(net, data, num_data, num_train):
    Z, U, X, Y = data
    Z = np.array(Z)
    U = np.array(U).T
    X = np.array(X).T
    Y = np.array(Y)

    Z_test = Z[num_train:]
    U_test = U[num_train:]
    Y_test = Y[num_train:]

    # PyTorch stuff
    with torch.no_grad():
        X_ = torch.from_numpy(X).float()
        Y_ = torch.from_numpy(np.array(Y).reshape((-1, 1))).float()

        num_test = num_data - num_train
        X_test = X_[num_train:]

        # Compute and print test accuracy
        Yhat = net(X_test).numpy().flatten()
        predictions = (Yhat > 0.5)
        correct = (predictions == Y_test).sum()
        accuracy = correct / num_test
        print('Accuracy: %d%%\n' % (100 * accuracy))

        # Extract intermediate activations from the network
        Xint = [actvn.numpy() for actvn in net.activations]

    num_layers = len(Xint)
    layer_sizes = [(Xint_.shape[1] if Xint_.ndim > 1 else 1) for Xint_ in Xint]

    z_corr = [corrcoef(Z_test[:, None], Xint[i]) for i in range(num_layers)]
    print('Correlations with Z:')
    for corr in z_corr: print(corr)
    print()

    print('Correlations with Y:')
    y_corr = [corrcoef(Y_test[:, None], Xint[i]) for i in range(num_layers)]
    for corr in y_corr: print(corr)
    print()

    print('Weights')
    weights = [getattr(net, 'fc%d' % i).weight.data.numpy()
               for i in range(1, num_layers)]
    print_edge_data(weights, layer_sizes)
    print()

    header = 'Layer\tX1\tX2\tX3\tX12\tX13\tX23\tX123'

    print('Computing bias flows...')
    print('Accuracies:')
    print(header)
    z_mis = [None,] * num_layers
    for i in range(num_layers):
        print(i, end='', flush=True)
        z_mis[i] = {(): 0}
        if Xint[i].ndim == 1:
            Xint[i] = Xint[i].reshape((-1, 1))
        for js in powerset(range(layer_sizes[i]), start=1):
            z_mi, z_acc = mutual_info_bin(Z_test, Xint[i][:, js], Hx=1,
                                          return_acc=True)
            z_mis[i][js] = z_mi
            print('\t%.4f' % z_acc, end='', flush=True)
        print()
    print('Mutual informations:')
    print_mis(z_mis, layer_sizes, header)
    print('Information flows:')
    z_info_flows = compute_all_flows(z_mis, layer_sizes)
    print_node_data(z_info_flows, layer_sizes)
    print('Weighted information flows:')
    z_info_flows_weighted = weight_info_flows(z_info_flows, weights)
    print_edge_data(z_info_flows_weighted, layer_sizes)
    print()

    print('Computing accuracy flows...')
    print('Accuracies:')
    print(header)
    y_mis = [None,] * num_layers
    for i in range(num_layers):
        print(i, end='', flush=True)
        y_mis[i] = {(): 0}
        if Xint[i].ndim == 1:
            Xint[i] = Xint[i].reshape((-1, 1))
        for js in powerset(range(layer_sizes[i]), start=1):
            y_mi, y_acc = mutual_info_bin(Y_test, Xint[i][:, js], Hx=1,
                                          return_acc=True)
            y_mis[i][js] = y_mi
            print('\t%.4f' % y_acc, end='', flush=True)
        print()
    print('Mutual informations:')
    print_mis(y_mis, layer_sizes, header)
    print('Information flows:')
    y_info_flows = compute_all_flows(y_mis, layer_sizes)
    print_node_data(y_info_flows, layer_sizes)
    print('Weighted information flows:')
    y_info_flows_weighted = weight_info_flows(y_info_flows, weights)
    print_edge_data(y_info_flows_weighted, layer_sizes)
    print()

    #plt.figure()
    #mask = (Yhat > 0.5)
    #plt.plot(U_test[mask, 0], U_test[mask, 1], 'C0o', alpha=0.3)
    #plt.plot(U_test[~mask, 0], U_test[~mask, 1], 'C1o', alpha=0.3)

    #plt.figure()
    #mask = (Z_test > 0.5)
    #plt.plot(U_test[mask, 0], Xint[2][mask, 0], 'C0o', alpha=0.3)
    #plt.plot(U_test[~mask, 0], Xint[2][~mask, 0], 'C1o', alpha=0.3)

    #plt.show()

    return (z_mis, z_info_flows, z_info_flows_weighted,
            y_mis, y_info_flows, y_info_flows_weighted,
            accuracy)


def prune_edge(net, layer, i, j, prune_factor=0):
    """
    Prunes an edge from neuron `i` to neuron `j` at a given `layer` in `net`
    by `prune_factor` and returns a copy.
    """

    pruned_net = SimpleNet()
    pruned_net.load_state_dict(net.state_dict())
    weights = copy.deepcopy(net.get_weights())
    weights[layer][i, j] *= prune_factor
    pruned_net.set_weights(weights)

    return pruned_net


if __name__ == '__main__':
    num_data = 2000
    #force_regenerate = False
    force_regenerate = True
    datafile = 'data-%d.pkl' % num_data
    # Load data if it exists; generate and save data otherwise
    if force_regenerate or not os.path.isfile(datafile):
        data = generate_data(num_data)
        print('Data generation complete')
        joblib.dump(data, datafile, compress=3)
    else:
        data = joblib.load(datafile)

    num_train = 1000
    annfile = 'nn-state.sav'
    net = train_ann(data, num_data, num_train, test=False, savefile=annfile)
    #net = SimpleNet()
    #net.load_state_dict(torch.load(annfile))

    #num_layers = 3
    #layer_sizes = [3, 3, 1]
    #print_edge_data(net.get_weights(), layer_sizes)
    #print()
    #pruned_net = prune_edge(net, 1, 0, 1)
    #pruned_net = prune_edge(pruned_net, 0, 2, 2, prune_factor=0.5)
    #pn_weights = [getattr(pruned_net, 'fc%d' % i).weight.data.numpy()
    #              for i in range(1, num_layers)]
    #print_edge_data(pn_weights, layer_sizes)
    #print()
    #print_edge_data(net.get_weights(), layer_sizes)

    #ret = analyze_info_flow(net, data, num_data, num_train)
    #(z_mis, z_info_flows, z_info_flows_weighted,
    # y_mis, y_info_flows, y_info_flows_weighted, acc_before) = ret
    #plot_ann(net.layer_sizes, z_info_flows_weighted)
    #plt.title('Weighted flows before pruning')
    #plt.savefig('figures/before.png' % pf)
    #plt.close()

    accs = []
    rets = []
    biases = []
    pfs = np.linspace(0, 1, 11)
    for pf in pfs:
        #pf = 0.5  # Prune factor (0 = edge removal; 1 = no pruning)

        #print('------------------------------------------------')
        #print('After pruning')
        #print('------------------------------------------------\n')

        print('------------------------------------------------')
        print('Prune factor: %g' % pf)
        print('------------------------------------------------\n')

        pruned_net = prune_edge(net, 0, 0, 0, prune_factor=pf)
        pruned_net = prune_edge(pruned_net, 0, 1, 0, prune_factor=pf)
        pruned_net = prune_edge(pruned_net, 0, 2, 0, prune_factor=pf)
        pruned_net = prune_edge(pruned_net, 0, 0, 1, prune_factor=pf)
        pruned_net = prune_edge(pruned_net, 0, 1, 1, prune_factor=pf)
        pruned_net = prune_edge(pruned_net, 0, 2, 1, prune_factor=pf)

        ret = analyze_info_flow(pruned_net, data, num_data, num_train)
        (z_mis, z_info_flows, z_info_flows_weighted,
         y_mis, y_info_flows, y_info_flows_weighted, acc) = ret

        plot_ann(pruned_net.layer_sizes, z_info_flows_weighted)
        plt.title('Weighted flows after pruning')
        plt.savefig('figures/weighted-flows-pf%g.png' % pf)

        accs.append(acc)
        rets.append(ret)
        biases.append(acc_from_mi(z_mis[2][(0,)]))
        #plt.show()

    np.savez_compressed('acc-bias-tradeoff.npz', accs=np.array(accs),
                        biases=np.array(biases))
    joblib.dump(rets, 'all-outputs.pkl', compress=3)

    plt.figure()
    plt.plot(biases, accs)
    plt.savefig('figures/bias-acc-tradeoff.png')
