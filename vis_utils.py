import ipywidgets
from IPython.display import display

from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

from nn import train_ann
from analyze_info_flow import analyze_info_flow
from plot_utils import init_plots, plot_ann


def init_vis(data, params):
    vis = SimpleNamespace()
    widgets = SimpleNamespace()

    # Define all widgets and add them to the widgets namespace
    define_widgets(widgets, params)

    # Add observers for responding to events on these widgets
    add_observers(vis, widgets, data, params)

    # Display all elements in the right order
    display(widgets.dropdown_dataset)  # TODO: This should be the dataset selector

    # Initialize plots
    # Under %matplotlib notebook, these show up immediately
    init_plots(vis)
    #reset_plots(vis, data, params)  # This actually does the initial plotting

    # Display text boxes and buttons
    display(widgets.box_prune)

    return vis, widgets


def define_widgets(widgets, params):
    # Dropdown for selecting a dataset
    widgets.dropdown_dataset = ipywidgets.Dropdown(
        options=params.datasets,
        value=params.default_dataset, description='Dataset:'
    )

    # Textbox for layer
    widgets.inttxt_layer = ipywidgets.IntText(description='Prune layer')
    # Textbox for from-node
    widgets.inttxt_from = ipywidgets.IntText(description='Prune from')
    # Textbox for to-node
    widgets.inttxt_to = ipywidgets.IntText(description='Prune to')
    # Textbox to adjust pruning factor
    widgets.fltxt_prune_factor = ipywidgets.FloatText(description='Prune factor')

    # Button for training the ANN
    widgets.button_train = ipywidgets.Button(description='Train ANN')
    # Button to calculate info flows
    widgets.button_calc_flow = ipywidgets.Button(description='Compute Info Flow')
    # Button for pruning
    widgets.button_prune = ipywidgets.Button(description='Prune')

    # Create a box to group all pruning elements
    widgets.box_prune = ipywidgets.HBox([
        ipywidgets.VBox([
            widgets.button_train,
            widgets.button_calc_flow,
        ]),
        ipywidgets.VBox([
            widgets.inttxt_layer,
            widgets.inttxt_from,
            widgets.inttxt_to,
            widgets.fltxt_prune_factor,
            widgets.button_prune,
        ]),
    ])


def add_observers(vis, widgets, data, params):
    # Update dropdown box to update dataset
    def dropdown_dataset_update(*args):
        init_data(widgets.dropdown_dataset.value, params, data)
    widgets.dropdown_dataset.observe(dropdown_dataset_update, 'value')

    def button_train_click(b):
        data.net = train_ann(data.data, params.num_data, params.num_train,
                             test=False, savefile=params.annfile)
        plot_ann(data.net.layer_sizes, data.net.get_weights(), ax=vis.ax_weights)
    widgets.button_train.on_click(button_train_click)

    # Update weights when prune button is clicked
    def button_calc_flow_click(b):
        ret = analyze_info_flow(data.net, data.data, params.num_data, params.num_train)
        (z_mis, z_info_flows, z_info_flows_weighted,
         y_mis, y_info_flows, y_info_flows_weighted) = ret
        plot_ann(data.net.layer_sizes, z_info_flows_weighted, ax=vis.ax_bias_flows)
        plot_ann(data.net.layer_sizes, y_info_flows_weighted, ax=vis.ax_acc_flows)
    widgets.button_calc_flow.on_click(button_calc_flow_click)

    # Update weights when prune button is clicked
    def button_prune_click(b):
        pass
    widgets.button_prune.on_click(button_prune_click)

