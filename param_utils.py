from types import SimpleNamespace


# Initialize parameters
def init_params(params=None):
    # Create a new namespace only if one has not already been provided
    if params is None:
        params = SimpleNamespace()

    # Data parameters
    params.datasets = ['TinySCM',]
    params.default_dataset = params.datasets[0]
    params.num_data = 2000
    params.num_train = 1000
    # Set datafile to None to regenerate data
    #params.datafile = 'data-%d.pkl' % params.num_data
    params.datafile = None
    # Set annfile to None to train the network from scratch
    #params.annfile = 'nn-state.sav'
    params.annfile = None

    return params
