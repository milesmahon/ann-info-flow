#!/usr/bin/env python3

from __future__ import print_function, division

import os
import sys
import joblib
import argparse
import numpy as np

from param_utils import init_params
from data_utils import init_data, print_data_stats
from pruning import prune
from analyze_info_flow import analyze_info_flow
from info_measures import acc_from_mi

from nn import SimpleNet  # Required for joblib.load to work - weird bug
# https://stackoverflow.com/questions/49621169/joblib-load-main-attributeerror


def concatenate(params):
    """
    Concatenate files from parallel runs
    """
    # XXX: This doesn't work currently
    raise NotImplementedError
    #filename, extension = params.analysis_file.rsplit('.', 1)

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

    parser = argparse.ArgumentParser(description='Analyze information flow and remove bias by pruning trained ANNs')
    parser.add_argument('-d', '--dataset', choices=params.datasets, help='Dataset to use for analysis')
    parser.add_argument('--metric', choices=params.prune_metrics, help='Choice of pruning metric')
    parser.add_argument('--method', choices=params.prune_methods, help='Choice of pruning method')
    parser.add_argument('--pruneamt', help='Amount by which to prune')
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
    if args.metric:
        params.prune_metric = args.metric
    if args.method:
        params.prune_method = args.method
    if args.pruneamt:
        params.num_to_prune = int(args.pruneamt)
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

    # NOTE: If the number of runs is >1, and params.force_retrain or
    # params.force_reanalyze are False, then the code will expect to find at least
    # `runs` instances of trained and/or analyzed nets in the respective files.

    # For now, keep data fixed across runs
    data = joblib.load(params.procdatafile)#init_data(params)
    print(params.num_data, params.num_train)
    #print_data_stats(data, params) # Check data statistics
    print(len(data))

    # Load all nets
    nets = joblib.load(params.annfile)

    # Load analyzed info flow data
    rets_before = joblib.load(params.analysis_file)

    accs = [[] for _ in range(len(runs_to_run))]
    rets = [[] for _ in range(len(runs_to_run))]
    biases = [[] for _ in range(len(runs_to_run))]
    pfs = params.prune_factors
    for run_index, run in enumerate(runs_to_run):
        print('------------------')
        print('Run %d' % run)

        # If the neural net save file or the analysis save file do not have a
        # sufficient number of runs, the following two lines will throw an error
        net = nets[run]
        ret_before = rets_before[run]
        for pf in pfs:
            print('------')
            print('Prune factor: %g' % pf)

            pruned_net = prune(net, ret_before[1], ret_before[4],
                               prune_factor=pf, params=params)

            ret = analyze_info_flow(pruned_net, data[run], params, full=False)
            z_mi, y_mi = ret

            rets[run_index].append(ret)

            # Raw accuracy from the neural network itself
            #accs.append(acc)
            # Accuracy based applying an SVM on top of the ANN's output
            accs[run_index].append(acc_from_mi(y_mi))

            biases[run_index].append(acc_from_mi(z_mi))

        # Append accuracies/biases for the before-pruning case
        rets[run_index].append(ret_before)
        accs[run_index].append(acc_from_mi(ret_before[3][-1][(0, 1)]))
        biases[run_index].append(acc_from_mi(ret_before[0][-1][(0, 1)]))
        #accs[run_index].append(acc_from_mi(max(ret_before[3][-1].values())))
        #biases[run_index].append(acc_from_mi(max(ret_before[0][-1].values())))
    print()

    job_suffix = ('-%d' % args.job) if args.job is not None else ''
    file_params = (params.prune_metric, params.prune_method,
                   params.num_to_prune, job_suffix)

    tradeoff_filename = os.path.join('results-%s' % params.dataset, args.subfolder,
                                     'tradeoff-%s-%s-%d%s.npz' % file_params)
    np.savez_compressed(tradeoff_filename, accs=np.array(accs),
                        biases=np.array(biases), prune_factors=pfs)

    rets_filename = os.path.join('results-%s' % params.dataset, args.subfolder,
                                 'rets-%s-%s-%d%s.pkl' % file_params)
    joblib.dump(dict(rets=rets, params=params), rets_filename, compress=3)
