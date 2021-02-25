import numpy as np
from types import SimpleNamespace
import torch.nn as nn


# Initialize parameters
def init_params(params=None):
    # Create a new namespace only if one has not already been provided
    if params is None:
        params = SimpleNamespace()

    # Data parameters
    params.datasets = ['tinyscm', 'adult']
    params.dataset = params.datasets[0]

    # Parameters specific to tinyscm
    params.num_data = 10000  # Should be moved to data.num_data etc at some point
    params.num_train = 5000
    params.datafile = 'results-%s/data-%d.pkl' % (params.dataset, params.num_data)
    params.force_regenerate = False # For simulated dataset
    #params.force_regenerate = True # For simulated dataset

    # ANN parameters
    params.annfile = 'results-%s/trained-nets.pkl' % params.dataset
    params.force_retrain = False

    # ANN training parameters for each dataset
    params.num_epochs = {'tinyscm': 50, 'adult': 50}
    params.minibatch_size = {'tinyscm': 10, 'adult': 10}  # Should be a factor of num_train for each dataset
    params.learning_rate = {'tinyscm': 0.03, 'adult': 3e-3}
    params.momentum = {'tinyscm': 0.9, 'adult': 0.9}
    params.print_every_factor = {'tinyscm': 5, 'adult': 5}  # Prints more for larger numbers
    params.criterion = {
        #'tinyscm': nn.MSELoss(),
        'tinyscm': nn.CrossEntropyLoss(),  # expects 1-hot encoding at output of NN
        #'adult': nn.MSELoss(),
        'adult': nn.CrossEntropyLoss(),  # expects 1-hot encoding at output of NN
    }

    # Parameters for initial analysis of the ANN
    params.analysis_file = 'results-%s/analyzed-data.pkl' % params.dataset
    params.force_reanalyze = False

    # Parameters for pruning
    params.prune_metrics = ['biasacc', 'accbias']
    params.prune_methods = ['node', 'edge', 'edge-rwf']
    params.prune_metric = params.prune_metrics[1]
    params.prune_method = params.prune_methods[1]
    params.num_to_prune = 2  # Number of nodes or edges to prune
    params.prune_factors = np.linspace(0, 1, 10, endpoint=False)
    #params.prune_factors = [0, 0.1, 0.5]
    params.num_runs = 1  # Number of times to run the analysis

    return params
