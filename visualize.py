# This is a snippet of a Jupyter notebook, not to be executed separately.
# It is meant to be imported using * in the notebook itself, to keep the
# notebook clean, and to enable the use of an editor of your choice.

import ipywidgets

from param_utils import init_params
from data_utils import init_data
from vis_utils import init_vis

params = init_params()
data = init_data(params)
vis, widgets = init_vis(data, params)
