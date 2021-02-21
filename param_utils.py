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
    params.default_dataset = params.datasets[0]
    params.num_data = 2000  # Should be moved to data.num_data etc at some point
    params.num_train = 1000
    params.datafile = 'data-%d.pkl' % params.num_data
    params.force_regenerate = False # For simulated dataset

    # ANN parameters
    params.annfile = 'results/nn-state.sav'
    #params.force_retrain = True
    params.force_retrain = False

    # ANN training parameters for each dataset
    params.num_epochs = {'tinyscm': 10, 'adult': 50}
    params.minibatch_size = {'tinyscm': 10, 'adult': 10}  # Should be a factor of num_train for each dataset
    params.learning_rate = {'tinyscm': 0.1, 'adult': 3e-3}
    params.momentum = {'tinyscm': 0.9, 'adult': 0.9}
    params.print_every_factor = {'tinyscm': 10, 'adult': 5}  # Prints more for larger numbers
    params.criterion = {
        #'tinyscm': nn.MSELoss(),
        'tinyscm': nn.CrossEntropyLoss(),  # expects 1-hot encoding at output of NN
        #'adult': nn.MSELoss(),
        'adult': nn.CrossEntropyLoss(),  # expects 1-hot encoding at output of NN
    }

    # Parameters for initial analysis of the ANN
    params.analysis_file = 'results/analyzed-data.pkl'
    #params.force_reanalyze = True
    params.force_reanalyze = False

    # Parameters for pruning
    params.prune_metrics = ['biasacc', 'accbias']
    params.prune_methods = ['node', 'edge', 'edge-rwf']
    params.prune_metric = params.prune_metrics[1]
    params.prune_method = params.prune_methods[1]
    params.num_to_prune = 2  # Number of nodes or edges to prune
    params.prune_factors = np.linspace(0, 1, 10, endpoint=False)
    #params.prune_factors = [0, 0.1, 0.5]

    return params
