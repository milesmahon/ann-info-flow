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
    params.annfile = 'nn-state.sav'
    params.force_retrain = False

    # ANN training parameters for each dataset
    params.num_epochs = {'tinyscm': 10, 'adult': 50}
    params.minibatch_size = {'tinyscm': 10, 'adult': 10}  # Should be a factor of num_train for each dataset
    params.learning_rate = {'tinyscm': 0.1, 'adult': 1e-2}
    params.momentum = {'tinyscm': 0.9, 'adult': 0.9}
    params.print_every_factor = {'tinyscm': 10, 'adult': 4}  # Prints more for larger numbers
    params.criterion = {
        #'tinyscm': nn.MSELoss(),
        'tinyscm': nn.CrossEntropyLoss(),  # expects 1-hot encoding at output of NN
        #'adult': nn.MSELoss() ,
        'adult': nn.CrossEntropyLoss(),  # expects 1-hot encoding at output of NN
    }

    return params
