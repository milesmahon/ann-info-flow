#!/usr/bin/env python3

from __future__ import print_function, division

import os
import sys
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from datasets.MotionColorDataset import MotionColorDataset
from param_utils import init_params
from data_utils import init_data, print_data_stats
from info_measures import (mutual_info_bin, compute_all_flows,
                           weight_info_flows, acc_from_mi)
from utils import powerset, print_mis, print_edge_data, print_node_data
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


def compute_info_flows(Z_test, Xint, layer_sizes, header, weights, full=True,
                       info_method=None, verbose=True):
    """
    Compute all information flows.

    If `full` is False, compute only mutual information on the final layer.
    """

    num_layers = len(layer_sizes)

    if full is not True:
        z_mi, z_acc = mutual_info_bin(Z_test, Xint[-1], Hx=1, return_acc=True, method=info_method)
        if verbose: print('Accuracy: %.4f' % z_acc)
        return z_mi

    if verbose:
        print('Accuracies:')
        print(header)

    z_mis = [None, ] * num_layers
    for i in range(num_layers):
        if verbose: print(i, end='', flush=True)
        z_mis[i] = {(): 0}
        if Xint[i].ndim == 1:
            Xint[i] = Xint[i].reshape((-1, 1))
        for js in powerset(range(layer_sizes[i]), start=1):  # start=1 avoids the empty set
            z_mi, z_acc = mutual_info_bin(Z_test, Xint[i][:, js], Hx=1,
                                          return_acc=True, method=info_method)
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


def analyze_info_flow(net, data, params, full=True, test=True):
    """
    Compute bias and accuracy flows on all edges of the neural net.

    If `full` is False, return bias and accuracy only for the final layer of
    the neural net (i.e., don't analyze flows).

    If `test` is False, perform info flow analysis on training data (should be
    used only for debugging)
    """
    X, Y, Z = data.data[:3]
    X = np.array(X)
    Y = np.array(Y)
    if 'adult' in data.dataset: # Adult dataset: 0 for race; 1 for gender
        Z = np.array(Z)[:, 1]
    else:                       # Others (incl Tiny SCM): only one protected attr
        Z = np.array(Z)

    if test:
        Z_test = Z[params.num_train:]
        Y_test = Y[params.num_train:]
    else:
        Z_test = Z[:params.num_train]
        Y_test = Y[:params.num_train]

    # PyTorch stuff
    with torch.no_grad():
        X_ = torch.from_numpy(X).float()

        if test:
            X_test = X_[params.num_train:]
            num_test = params.num_data - params.num_train
        else:
            X_test = X_[:params.num_train]
            num_test = params.num_train

        # Compute and print test accuracy
        net.eval()
        Yhat = np.squeeze(net(X_test).numpy())
        #predictions = (Yhat > 0.5)  # Logic for single output node encoding 0/1 using a sigmoid
        predictions = (Yhat[:, 0] < Yhat[:, 1]).astype(int)  # Logic for 1-hot encoding of 0/1 at output node
        correct = (predictions == Y_test).sum()
        accuracy = correct / num_test

        print('Accuracy: %.5g%%' % (100 * accuracy))
        if full: print()

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
        ret = compute_info_flows(Z_test, Xint, layer_sizes, header, weights,
                                 full=full, info_method=params.info_method, verbose=True)
        z_mis, z_info_flows, z_info_flows_weighted = ret
    else:
        z_mi = compute_info_flows(Z_test, Xint, layer_sizes, header, weights,
                                  full=full, info_method=params.info_method, verbose=True)

    print('Computing accuracy flows...')
    if full:
        ret = compute_info_flows(Y_test, Xint, layer_sizes, header, weights,
                                 full=full, info_method=params.info_method, verbose=True)
        y_mis, y_info_flows, y_info_flows_weighted = ret
    else:
        y_mi = compute_info_flows(Y_test, Xint, layer_sizes, header, weights,
                                  full=full, info_method=params.info_method, verbose=True)

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


# from RNN model output, return -1, 0 or 1
# def translate_output(x):
#     classes = [-1, 0, 1]
#     prob = nn.functional.softmax(x[-1], dim=0).data
#     choice = classes[torch.max(prob, dim=0)[1].item()]
#     return choice


# X is inputs
# Y is true labels
# Z is input to ignore
# could also be interesting to track info flow of context, especially if only provided once.
    # vanishing gradient problem if context is provided only once (either at the beginning or end of the sequence)
# could train on both color and motion, test on just motion, see if color is represented
# could also think of X as inputs, Y = motion Z = color (labels, that is one-hot encodings of the mean of the
    # distributions representing them).
# e.g. X = [1.22, -1.05, 1.0] (motion, color, context)
    # Y = [0, 0, 1] (one-hot encode of 1, or "right" motion)
    # Z = [1, 0, 0] (one-hot encode of -1, or "red" color)
def analyze_info_flow_rnn(net, info_method, full=True, test=True):
    """
    Compute bias and accuracy flows on all edges of the RNN.

    If `full` is False, return bias and accuracy only for the final layer of
    the neural net (i.e., don't analyze flows).

    If `test` is False, perform info flow analysis on training data (should be
    used only for debugging)
    """
    num_data = 2000
    num_train = 1000  # must be = batch size
    context_time = "retro"
    vary_acc = True  # TODO if set desired_acc, set vary_acc to false
    figure_filename = "figs/MCCTrain99Des85Retro.png"

    mc_dataset = MotionColorDataset(num_data, 10)  # TODO pass dataset from training
    X, Y, Z, true_labels, C = mc_dataset.get_xyz(num_data, context_time=context_time, vary_acc=vary_acc)
    X = np.array(X)  # input
    Y = np.array(Y)  # color
    Z = np.array(Z)  # motion
    U = np.array(true_labels)  # true label
    C = np.array(C)

    # PyTorch stuff
    with torch.no_grad():
        X_test = X[num_train:]
        Y_test = Y[num_train:]
        Z_test = Z[num_train:]
        U_test = U[num_train:]
        C_test = C[num_train:]
        num_test = num_data - num_train

        # Compute and print test accuracy
        net.eval()
        correct = 0
        # for i in range(num_test):
        hidden = net.init_hidden()
        output, hidden = net(torch.from_numpy(X_test).float(), hidden.float())
        # Extract intermediate activations from the network
        Xint = [actvn.numpy() for actvn in net.activations]

        Yhat = np.squeeze(output.numpy())
        #predictions = (Yhat > 0.5)  # Logic for single output node encoding 0/1 using a sigmoid
        predictions = (Yhat[:, 0] < Yhat[:, 1]).astype(int)  # Logic for 1-hot encoding of 0/1 at output node
        correct = (predictions == U_test).sum()
        accuracy = correct / num_test

        print('Accuracy: %.5g%%' % (100 * accuracy))
        if full:
            print()

        #num_layers = len(Xint)
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

        # time unroll weights
        # TODO MM hardcoded for now since dependent on layout of network
        new_weights = []
        # new_weights.append(weights[0])  # ih
        for i in range(9):
            new_weights.append(weights[1])
        new_weights.append(weights[2])  # fc layer
        weights = new_weights

        # time unroll RNN activations
        new_layers = []
        for layer in Xint[:-1]:  # trials x seq length x activations per layer. exclude last fc layer
            for i in range(layer.shape[1]):  # seq. length
                new_layers.append(layer[:, i])  # activations in sequence are separate layers
        new_layers.append(np.squeeze(Xint[-1])) # fc layer
        Xint = new_layers
        layer_sizes = [(Xint_.shape[1] if Xint_.ndim > 1 else 1) for Xint_ in Xint]

        header = 'Layer\tX1\tX2\tX3\tX12\tX13\tX23\tX123'

        print('Computing motion flows...')
        if full:
            ret = compute_info_flows(Z_test, Xint, layer_sizes, header, weights,
                                     full=full, info_method=info_method, verbose=True)
            z_mis, z_info_flows, z_info_flows_weighted = ret
        else:
            z_mi = compute_info_flows(Z_test, Xint, layer_sizes, header, weights,
                                      full=full, info_method=info_method, verbose=True)

        print('Computing color flows...')
        if full:
            ret = compute_info_flows(Y_test, Xint, layer_sizes, header, weights,
                                     full=full, info_method=info_method, verbose=True)
            y_mis, y_info_flows, y_info_flows_weighted = ret
        else:
            y_mi = compute_info_flows(Y_test, Xint, layer_sizes, header, weights,
                                      full=full, info_method=info_method, verbose=True)

        # if context_time != "retro":  # if retrospective context, no need to get context flow
        print('Computing context flows...')
        if full:
            ret = compute_info_flows(C_test, Xint, layer_sizes, header, weights,
                                     full=full, info_method=info_method, verbose=True)
            c_mis, c_info_flows, c_info_flows_weighted = ret
        else:
            c_mi = compute_info_flows(C_test, Xint, layer_sizes, header, weights,
                                      full=full, info_method=info_method, verbose=True)

        unity_weights = [np.ones_like(w) for w in weights]
        flows = [abs(y) for y in weight_info_flows(y_info_flows, unity_weights)]
        plot_ann(layer_sizes, flows, flow_type='acc', label_name='Unweighted color flow in RNN')

        flows = [abs(z) for z in weight_info_flows(z_info_flows, unity_weights)]
        plot_ann(layer_sizes, flows, flow_type='bias', label_name='Unweighted motion flow in RNN')

        # if context_time != "retro":
        flows = [abs(c) for c in weight_info_flows(c_info_flows, unity_weights)]
        plot_ann(layer_sizes, flows, flow_type='context', label_name='Unweighted context flow in RNN')

        weights = [abs(c) for c in c_info_flows_weighted]
        plot_ann(layer_sizes, weights, flow_type='context', label_name='Weighted context flow in RNN')

        weights = [abs(w) for w in y_info_flows_weighted]
        plot_ann(layer_sizes, weights, flow_type='acc', label_name='Weighted color flow in RNN')

        weights = [abs(w) for w in z_info_flows_weighted]
        plot_ann(layer_sizes, weights, flow_type='bias', label_name='Weighted motion flow in RNN')
        plt.show()
        # plt.savefig(figure_filename)

        print("Done")

        # TODO MM return c info flows?
        if full:
            return (z_mis, z_info_flows, z_info_flows_weighted,
                    y_mis, y_info_flows, y_info_flows_weighted,
                    accuracy)
        else:
            return z_mi, y_mi, c_mi


def concatenate(params):
    """
    Concatenate files from parallel runs
    """
    filename, extension = params.analysis_file.rsplit('.', 1)

    rets_before = []
    for run in range(params.num_runs):
        job_suffix = '-%d' % run
        savefile = '.'.join([filename + job_suffix, extension])
        rets_part = joblib.load(savefile)
        rets_before.extend(rets_part)

    joblib.dump(rets_before, params.analysis_file, compress=3)


if __name__ == '__main__':
    # The params defined below is used for setting argument choices, and default
    # parameters if dataset is not given in arguments
    params = init_params()

    parser = argparse.ArgumentParser(description='Analyze information flow analysis on trained ANNs')
    parser.add_argument('-d', '--dataset', choices=params.datasets, help='Dataset to use for analysis')
    parser.add_argument('--runs', type=int, help='Number of times to run the analysis.')
    parser.add_argument('-j', '--job', type=int, default=None,
                        help='Job number (in 0 .. runs-1): when set, parallelizes over runs; expects `runs` number of jobs')
    parser.add_argument('--concatenate', action='store_true', help='Concatenate files from parallel runs and exit')
    parser.add_argument('--info-method', choices=params.info_methods, default=None, help='Choice of information estimation method')
    parser.add_argument('--subfolder', default='', help='Subfolder for results')
    args = parser.parse_args()

    print('\n------------------------------------------------')
    print(args)
    print('------------------------------------------------\n')

    # Override params specified in param_utils, if given in CLI
    if args.dataset:
        # Reinitialize params if the dataset is given
        params = init_params(dataset=args.dataset)
    if args.runs:
        params.num_runs = args.runs
    if args.job is not None:
        if args.job < 0 or args.job >= params.num_runs:
            raise ValueError('`job` must be between 0 and `runs`-1')
        runs_to_run = [args.job,]
    else:
        runs_to_run = range(params.num_runs)
    if args.info_method is not None:
        params.info_method = args.info_method
    if args.subfolder:
        results_dir, filename = os.path.split(params.analysis_file)
        params.analysis_file = os.path.join(results_dir, args.subfolder, filename)
        os.makedirs(os.path.join(results_dir, args.subfolder), exist_ok=True)
    if args.concatenate:
        concatenate(params)
        sys.exit(0)

    # For now, keep data fixed across runs
    data = init_data(params)
    print(params.num_data, params.num_train)
    print_data_stats(data, params) # Check data statistics

    # Load all nets
    # If the number of runs is >1, the code will expect to find at least `runs`
    # instances of trained nets
    nets = joblib.load(params.annfile)

    # Analyze all nets
    rets_before = []
    for run in runs_to_run:
        print('------------------')
        print('Run %d' % run)
        # Must do a full analysis the first time around
        ret_before = analyze_info_flow(nets[run], data, params, full=True)
        rets_before.append(ret_before)

    filename, extension = params.analysis_file.rsplit('.', 1)
    job_suffix = ('-%d' % args.job) if args.job is not None else ''
    savefile = '.'.join([filename + job_suffix, extension])
    joblib.dump(rets_before, savefile, compress=3)

    #(z_mis, z_info_flows, z_info_flows_weighted,
    # y_mis, y_info_flows, y_info_flows_weighted, acc) = ret_before

    #plot_ann(net.layer_sizes, ret_before[2])
    #plt.title('Weighted flows before pruning')
    #plt.savefig('figures/weighted-flows-before.png')
    #plt.close()
