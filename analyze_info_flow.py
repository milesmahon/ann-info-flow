#!/usr/bin/env python3

from __future__ import print_function, division

import os
import copy
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch

from param_utils import init_params
from data_utils import init_data, generate_data
from info_measures import (mutual_info_bin, compute_all_flows,
                           weight_info_flows, acc_from_mi)
from pruning import prune_nodes_biasacc
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


def analyze_info_flow(net, data, params): #num_data, num_train):
    X, Y, Z = data.data[:3]
    X = np.array(X)
    Y = np.array(Y)
    if data.dataset == 'adult': # Adult dataset: 0 for race; 1 for gender
        Z = np.array(Z)[:, 1]
    else:                       # Others (incl Tiny SCM): only one protected attr
        Z = np.array(Z)

    Z_test = Z[params.num_train:]
    Y_test = Y[params.num_train:]

    # PyTorch stuff
    with torch.no_grad():
        X_ = torch.from_numpy(X).float()
        Y_ = torch.from_numpy(np.array(Y).reshape((-1, 1))).float()

        num_test = params.num_data - params.num_train
        X_test = X_[params.num_train:]

        # Compute and print test accuracy
        net.eval()
        Yhat = np.squeeze(net(X_test).numpy())
        #predictions = (Yhat > 0.5)  # Logic for single output node encoding 0/1 using a sigmoid
        predictions = (Yhat[:, 0] < Yhat[:, 1]).astype(int)  # Logic for 1-hot encoding of 0/1 at output node
        correct = (predictions == Y_test).sum()
        accuracy = correct / num_test

        print('Accuracy: %.5g%%\n' % (100 * accuracy))

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
        for js in powerset(range(layer_sizes[i]), start=1):  # start=1 avoids the empty set
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


if __name__ == '__main__':
    params = init_params()
    #params.force_regenerate = True  # Only relevant for tinyscm
    #params.force_retrain = True
    #params.force_reanalyze = True

    #data = init_data(params, dataset='tinyscm')
    data = init_data(params, dataset='adult')
    print(params.num_data, params.num_train)

    # Check data statistics
    X, Y, Z = data.data[:3]
    X = np.array(X)
    Y = np.array(Y)
    if data.dataset == 'adult': # Adult dataset: 0 for race; 1 for gender
        Z = np.array(Z)[:, 1]
    else:                       # Others (incl Tiny SCM): only one protected attr
        Z = np.array(Z)

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

    if params.force_retrain or params.annfile is None:
        net = train_ann(data, params, test=False, savefile=params.annfile)
    else:
        net = SimpleNet()
        net.load_state_dict(torch.load(params.annfile))

    # Analyze the network before pruning
    if params.force_reanalyze or params.analysis_file is None:
        ret_before = analyze_info_flow(net, data, params)
        joblib.dump(ret_before, params.analysis_file, compress=3)
    else:
        ret_before = joblib.load(params.analysis_file)
    #(z_mis, z_info_flows, z_info_flows_weighted,
    # y_mis, y_info_flows, y_info_flows_weighted, acc) = ret_before
    plot_ann(net.layer_sizes, ret_before[2])
    plt.title('Weighted flows before pruning')
    plt.savefig('figures/weighted-flows-before.png')
    plt.close()

    accs = []
    rets = []
    biases = []
    #pfs = [1,]
    pfs = np.linspace(0, 1, 10, endpoint=False)
    num_to_prune = 1
    for pf in pfs:
        print('------------------------------------------------')
        print('Prune factor: %g' % pf)
        print('------------------------------------------------\n')

        pruned_net = prune_nodes_biasacc(net, ret_before[1], ret_before[4],
                                         num_nodes=num_to_prune, prune_factor=pf)

        ret = analyze_info_flow(pruned_net, data, params)
        (z_mis, z_info_flows, z_info_flows_weighted,
         y_mis, y_info_flows, y_info_flows_weighted, acc) = ret

        plot_ann(pruned_net.layer_sizes, z_info_flows_weighted)
        plt.title('Prune factor: %g' % pf)
        plt.savefig('figures/weighted-flows-pf%g.png' % pf)

        rets.append(ret)

        #accs.append(acc)  # Raw accuracy from the neural network itself
        #accs.append(acc_from_mi(y_mis[2][(0,)]))  # If using a single output node
        accs.append(acc_from_mi(y_mis[2][(0, 1)]))  # If using a 1-hot encoded output

        #biases.append(acc_from_mi(z_mis[2][(0,)]))  # If using a single output node
        biases.append(acc_from_mi(z_mis[2][(0, 1)]))  # If using a 1-hot encoded output

    np.savez_compressed('results/acc-bias-tradeoff.npz', accs=np.array(accs),
                        biases=np.array(biases), prune_factors=pfs)
    joblib.dump(rets, 'results/all-outputs.pkl', compress=3)

    plt.figure()
    plt.plot(biases, accs, 'C0-o')
    plt.savefig('figures/bias-acc-tradeoff.png')
