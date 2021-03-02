#!/usr/bin/env python3

from __future__ import print_function, division

import sys
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from param_utils import init_params
from data_utils import init_data, generate_data
from info_measures import (mutual_info_bin, compute_all_flows,
                           weight_info_flows, acc_from_mi)
from pruning import prune
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


def compute_info_flows(Z_test, Xint, layer_sizes, header, weights, full=True, verbose=True):
    """
    Compute all information flows.

    If `full` is False, compute only mutual information on the final layer.
    """

    num_layers = len(layer_sizes)

    if full is not True:
        z_mi, z_acc = mutual_info_bin(Z_test, Xint[-1], Hx=1, return_acc=True)
        if verbose: print('Accuracy: %.4f' % z_acc)
        return z_mi

    if verbose:
        print('Accuracies:')
        print(header)

    z_mis = [None,] * num_layers
    for i in range(num_layers):
        if verbose: print(i, end='', flush=True)
        z_mis[i] = {(): 0}
        if Xint[i].ndim == 1:
            Xint[i] = Xint[i].reshape((-1, 1))
        for js in powerset(range(layer_sizes[i]), start=1):  # start=1 avoids the empty set
            z_mi, z_acc = mutual_info_bin(Z_test, Xint[i][:, js], Hx=1,
                                          return_acc=True)
            z_mis[i][js] = z_mi
            if verbose: print('\t%.4f' % z_acc, end='', flush=True)
        if verbose: print()

    z_info_flows = compute_all_flows(z_mis, layer_sizes)
    z_info_flows_weighted = weight_info_flows(z_info_flows, weights)

    if verbose:
        print('Mutual informations:')
        print_mis(z_mis, layer_sizes, header)
        print('Information flows:')
        print_node_data(z_info_flows, layer_sizes)
        print('Weighted information flows:')
        print_edge_data(z_info_flows_weighted, layer_sizes)
        print()

    return z_mis, z_info_flows, z_info_flows_weighted


def analyze_info_flow(net, data, params, full=True):
    """
    Compute bias and accuracy flows on all edges of the neural net.

    If `full` is False, return bias and accuracy only for the final layer of
    the neural net (i.e., don't analyze flows).
    """
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

    #z_corr = [corrcoef(Z_test[:, None], Xint[i]) for i in range(num_layers)]
    #print('Correlations with Z:')
    #for corr in z_corr: print(corr)
    #print()

    #print('Correlations with Y:')
    #y_corr = [corrcoef(Y_test[:, None], Xint[i]) for i in range(num_layers)]
    #for corr in y_corr: print(corr)
    #print()

    #print('Weights')
    weights = net.get_weights()
    #print_edge_data(weights, layer_sizes)
    #print()

    header = 'Layer\tX1\tX2\tX3\tX12\tX13\tX23\tX123'

    print('Computing bias flows...')
    if full:
        ret = compute_info_flows(Z_test, Xint, layer_sizes, header, weights, full=full, verbose=True)
        z_mis, z_info_flows, z_info_flows_weighted = ret
    else:
        z_mi = compute_info_flows(Z_test, Xint, layer_sizes, header, weights, full=full, verbose=True)

    print('Computing accuracy flows...')
    if full:
        ret = compute_info_flows(Y_test, Xint, layer_sizes, header, weights, full=full, verbose=True)
        y_mis, y_info_flows, y_info_flows_weighted = ret
    else:
        y_mi = compute_info_flows(Y_test, Xint, layer_sizes, header, weights, full=full, verbose=True)

    #plt.figure()
    #mask = (Yhat > 0.5)
    #plt.plot(U_test[mask, 0], U_test[mask, 1], 'C0o', alpha=0.3)
    #plt.plot(U_test[~mask, 0], U_test[~mask, 1], 'C1o', alpha=0.3)

    #plt.figure()
    #mask = (Z_test > 0.5)
    #plt.plot(U_test[mask, 0], Xint[2][mask, 0], 'C0o', alpha=0.3)
    #plt.plot(U_test[~mask, 0], Xint[2][~mask, 0], 'C1o', alpha=0.3)

    #plt.show()

    if full:
        return (z_mis, z_info_flows, z_info_flows_weighted,
                y_mis, y_info_flows, y_info_flows_weighted,
                accuracy)
    else:
        return z_mi, y_mi


def print_data_stats(data):
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


if __name__ == '__main__':
    params = init_params()
    # Set all default parameters in param_utils!

    parser = argparse.ArgumentParser(description='Information flow analysis and'
                                     +'bias removal by pruning on trained ANNs')
    parser.add_argument('-d', '--dataset', choices=params.datasets, help='Dataset to use for analysis')
    parser.add_argument('--metric', choices=params.prune_metrics, help='Choice of pruning metric')
    parser.add_argument('--method', choices=params.prune_methods, help='Choice of pruning method')
    parser.add_argument('--pruneamt', help='Amount by which to prune')
    parser.add_argument('--runs', type=int, help='Number of times to run the analysis.')
    parser.add_argument('--retrain', action='store_true', help='Retrain the ANNs if set')
    parser.add_argument('--reanalyze', action='store_true', help='Reanalyze the ANNs if set')
    parser.add_argument('--train-only', action='store_true', help='Stop after training')
    parser.add_argument('--analyze-only', action='store_true', help='Stop after analysis')
    args = parser.parse_args()

    print('\n------------------------------------------------')
    print(args)
    print('------------------------------------------------\n')

    # Override params specified in param_utils, if given in CLI
    if args.dataset:
        params.dataset = args.dataset
    if args.metric:
        params.prune_metric = args.metric
    if args.method:
        params.prune_method = args.method
    if args.pruneamt:
        params.num_to_prune = int(args.pruneamt)
    if args.runs:
        params.num_runs = args.runs
    if args.retrain:
        params.force_retrain = True
    if args.reanalyze:
        params.force_reanalyze = True

    # NOTE: If the number of runs is >1, and params.force_retrain or
    # params.force_reanalyze are False, then the code will expect to find at least
    # `runs` instances of trained and/or analyzed nets in the respective files.

    # For now, keep data fixed across runs
    data = init_data(params)
    print(params.num_data, params.num_train)
    print_data_stats(data) # Check data statistics

    # Train all neural nets in advance
    if params.force_retrain or params.annfile is None:
        nets = []
        for run in range(params.num_runs):
            print('------------------')
            print('Run %d' % run)
            net = train_ann(data, params, test=False, random_seed=(1000+run))
            nets.append(net)
        #torch.save(net.state_dict(), params.annfile)
        joblib.dump(nets, params.annfile, compress=3)
    else:
        #net = SimpleNet()
        #net.load_state_dict(torch.load(params.annfile))
        nets = joblib.load(params.annfile)
    if args.train_only:
        sys.exit(0)

    # Analyze all nets before pruning
    if params.force_reanalyze or params.analysis_file is None:
        rets_before = []
        for run in range(params.num_runs):
            print('------------------')
            print('Run %d' % run)
            ret_before = analyze_info_flow(nets[run], data, params, full=True)  # Must do a full analysis
            rets_before.append(ret_before)
        joblib.dump(rets_before, params.analysis_file, compress=3)
    else:
        rets_before = joblib.load(params.analysis_file)
    #(z_mis, z_info_flows, z_info_flows_weighted,
    # y_mis, y_info_flows, y_info_flows_weighted, acc) = ret_before
    if args.analyze_only:
        sys.exit(0)

    #plot_ann(net.layer_sizes, ret_before[2])
    #plt.title('Weighted flows before pruning')
    #plt.savefig('figures/weighted-flows-before.png')
    #plt.close()

    accs = [[] for _ in range(params.num_runs)]
    rets = [[] for _ in range(params.num_runs)]
    biases = [[] for _ in range(params.num_runs)]
    pfs = params.prune_factors
    for run in range(params.num_runs):
        print('------------------')
        print('Run %d' % run)

        # If the neural net save file or the analysis save file do not have a
        # sufficient number of runs, the following two lines will throw an error
        net = nets[run]
        ret_before = rets_before[run]
        for pf in pfs:
            print('------')
            print('Prune factor: %g' % pf)

            pruned_net = prune(net, ret_before[1], ret_before[4], prune_factor=pf,
                               params=params)

            # TODO: Important point: In order to make bias-acc tradeoff curves, we
            # don't need to reanalyze the whole network. It suffices to measure
            # the bias alone at the output. The rest is needed only for
            # visualization, which is not a major focus right now.
            #ret = analyze_info_flow(pruned_net, data, params, full=True)
            #(z_mis, z_info_flows, z_info_flows_weighted,
            # y_mis, y_info_flows, y_info_flows_weighted, acc) = ret
            ret = analyze_info_flow(pruned_net, data, params, full=False)
            z_mi, y_mi = ret

            #plot_ann(pruned_net.layer_sizes, z_info_flows_weighted)
            #plt.title('Prune factor: %g' % pf)
            #plt.savefig('figures/weighted-flows-pf%g.png' % pf)

            rets[run].append(ret)

            # Raw accuracy from the neural network itself
            #accs.append(acc)
            # Accuracy based applying an SVM on top of the ANN's output
            #accs.append(acc_from_mi(y_mis[2][(0,)]))  # If using a single output node
            #accs.append(acc_from_mi(y_mis[2][(0, 1)]))  # If using a 1-hot encoded output
            accs[run].append(acc_from_mi(y_mi))

            #biases.append(acc_from_mi(z_mis[2][(0,)]))  # If using a single output node
            #biases.append(acc_from_mi(z_mis[2][(0, 1)]))  # If using a 1-hot encoded output
            biases[run].append(acc_from_mi(z_mi))

        # Append accuracies/biases for the before-pruning case
        rets[run].append(ret_before)
        accs[run].append(acc_from_mi(ret_before[3][2][(0, 1)]))
        biases[run].append(acc_from_mi(ret_before[0][2][(0, 1)]))
    print()

    file_params = (params.dataset, params.prune_metric, params.prune_method,
                   params.num_to_prune)
    tradeoff_filename = ('results-%s/tradeoff-%s-%s-%d.npz' % file_params)
    np.savez_compressed(tradeoff_filename, accs=np.array(accs),
                        biases=np.array(biases), prune_factors=pfs)
    rets_filename = ('results-%s/rets-%s-%s-%d.pkl' % file_params)
    joblib.dump(dict(rets=rets, params=params), rets_filename, compress=3)
