#!/usr/bin/env python3

from __future__ import print_function, division

import sys
import joblib
import argparse
import numpy as np

from param_utils import init_params
from data_utils import init_data, print_data_stats
from pruning import edge_list, prune_edge
from analyze_info_flow import analyze_info_flow
from info_measures import acc_from_mi

from nn import SimpleNet  # Required for joblib.load to work
# https://stackoverflow.com/questions/49621169/joblib-load-main-attributeerror


def concatenate(params):
    """
    Concatenate files from parallel runs
    """
    delta_accs = []
    delta_biases = []
    for run in range(params.num_runs):
        job_suffix = '-%d' % run
        tradeoff_filename = ('results-%s/scaling%s.npz' % (params.dataset, job_suffix))
        data = np.load(tradeoff_filename)
        delta_accs.append(data['delta_accs'])
        delta_biases.append(data['delta_biases'])

    delta_accs = np.concatenate(delta_accs)
    delta_biases = np.concatenate(delta_biases)

    tradeoff_filename = ('results-%s/scaling.npz' % params.dataset)
    np.savez_compressed(tradeoff_filename, delta_accs=delta_accs, delta_biases=delta_biases)


if __name__ == '__main__':
    # The params defined below is used for setting argument choices, and default
    # parameters if dataset is not given in arguments
    params = init_params()

    parser = argparse.ArgumentParser(description='Scaling of intervention effect with info flow magnitude')
    parser.add_argument('-d', '--dataset', choices=params.datasets, help='Dataset to use for analysis')
    parser.add_argument('--runs', type=int, help='Number of times to run the analysis.')
    parser.add_argument('-j', '--job', type=int, default=None,
                        help='Job number (in 0 .. runs-1): when set, parallelizes over runs; expects `runs` number of jobs')
    parser.add_argument('--concatenate', action='store_true', help='Concatenate files from parallel runs and exit')
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
    if args.concatenate:
        concatenate(params)
        sys.exit(1)

    # For now, keep data fixed across runs
    data = init_data(params)
    print(params.num_data, params.num_train)
    print_data_stats(data, params) # Check data statistics

    # Load all nets
    nets = joblib.load(params.annfile)

    # Load analyzed info flow data
    rets_before = joblib.load(params.analysis_file)

    # Analyze how the magnitude of information flow on a particular edge
    # scales with the effect of intervening on that edge
    orig_accs = []
    orig_biases = []
    delta_accs = [[] for _ in range(len(runs_to_run))]
    delta_biases = [[] for _ in range(len(runs_to_run))]
    bias_flows = []
    acc_flows = []
    for run_index, run in enumerate(runs_to_run):
        print('------------------')
        print('Run %d' % run)

        # If the neural net save file or the analysis save file do not have a
        # sufficient number of runs, the following two lines will throw an error
        net = nets[run]
        ret_before = rets_before[run]

        orig_bias = acc_from_mi(ret_before[0][-1][(0, 1)])
        orig_acc = acc_from_mi(ret_before[3][-1][(0, 1)])
        orig_biases.append(orig_bias)
        orig_accs.append(orig_acc)

        edges, bias_flow = edge_list(net, ret_before[2])
        _, acc_flow = edge_list(net, ret_before[5])
        bias_flows.append(bias_flow)
        acc_flows.append(acc_flow)

        for edge in edges:
            print(edge)
            pruned_net = prune_edge(net, *edge)
            ret = analyze_info_flow(pruned_net, data, params, full=False)
            z_mi, y_mi = ret
            delta_biases[run_index].append(acc_from_mi(z_mi) - orig_bias)
            delta_accs[run_index].append(acc_from_mi(y_mi) - orig_acc)
            print()

    job_suffix = ('-%d' % args.job) if args.job is not None else '' # For savefiles
    tradeoff_filename = ('results-%s/scaling%s.npz' % (params.dataset, job_suffix))
    np.savez_compressed(tradeoff_filename, orig_accs=np.array(orig_accs),
                        orig_biases=np.array(orig_biases),
                        delta_accs=np.array(delta_accs),
                        delta_biases=np.array(delta_biases),
                        acc_flows=np.array(acc_flows),
                        bias_flows=np.array(bias_flows))
