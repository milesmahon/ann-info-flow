#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import copy
from nn import SimpleNet
from info_measures import weight_info_flows


def prune_edge(net, layer, i, j, prune_factor=0, return_copy=True):
    """
    Prunes an edge from neuron `i` to neuron `j` at a given `layer` in `net`
    by `prune_factor` and returns a copy (if `return_copy` is True).
    """

    # Create a copy if required
    if return_copy:
        pruned_net = SimpleNet()
        pruned_net.load_state_dict(net.state_dict())
    else:
        pruned_net = net

    weights = copy.deepcopy(pruned_net.get_weights())
    weights[layer][j, i] *= prune_factor
    pruned_net.set_weights(weights)

    return pruned_net


def prune_node(net, layer, i, prune_factor=0, return_copy=True):
    """
    Reduce the weights of all outgoing edges of a node by `prune_factor`.
    """

    # Create a copy if required
    if return_copy:
        pruned_net = SimpleNet()
        pruned_net.load_state_dict(net.state_dict())
    else:
        pruned_net = net

    if layer == net.num_layers - 1:
        # If `layer` is the last layer, then there's nothing to do;
        return pruned_net
    else:
        for j in range(net.layer_sizes[layer + 1]):
            pruned_net = prune_edge(pruned_net, layer, i, j,
                                    prune_factor=prune_factor, return_copy=False)

    return pruned_net


def prune_nodes_biasacc(net, z_info_flows, y_info_flows, num_nodes=1, prune_factor=0, accbias=False):
    """
    Prune nodes in decreasing order of bias-to-accuracy or accuracy-to-bias ratio.
    """

    # Compute node scores by bias accuracy ratio
    bias_acc_ratio = [None,] * net.num_layers
    nodes = []
    node_scores = []
    for k in range(net.num_layers - 1): # Layer is indexed by 'k'; ignore last layer
        if accbias:
            bias_acc_ratio[k] = np.array(y_info_flows[k]) / np.array(z_info_flows[k])
        else:
            bias_acc_ratio[k] = np.array(z_info_flows[k]) / np.array(y_info_flows[k])

        for i in range(net.layer_sizes[k]): # Node within layer is indexed by 'i'
            nodes.append([k, i])
            node_scores.append(bias_acc_ratio[k][i])

    nodes = np.array(nodes)
    node_scores = np.array(node_scores)

    # Sort nodes by descending order of node score
    sort_inds = np.argsort(node_scores)[::-1]
    nodes = nodes[sort_inds]

    # Prune up to `num_nodes` by prune_factor
    pruned_net = SimpleNet()
    pruned_net.load_state_dict(net.state_dict())
    for k, i in nodes[:num_nodes]:
        pruned_net = prune_node(pruned_net, k, i, prune_factor=prune_factor,
                                return_copy=False)

    return pruned_net


def prune_edges_proportional(net, z_info_flows, y_info_flows, num_edges=1, prune_factor=0, accbias=False):
    """
    Prune edges in decreasing order of weighted bias-to-accuracy or weighted
    accuracy-to-bias ratio.
    """

    # Compute biasacc or accbias ratio, and weight it
    ratio = [None,] * net.num_layers
    for k in range(net.num_layers):
        if accbias:
            ratio[k] = y_info_flows[k] / z_info_flows[k]
        else:
            ratio[k] = z_info_flows[k] / y_info_flows[k]
    ratio_weighted = weight_info_flows(ratio, net.get_weights())
    # Note: weighted flows are still signed! Need to take abs() below.

    # Vectorize weighted ratios to create edge scores
    edges = []
    edge_scores = []
    for k in range(net.num_layers - 1):
        for i in range(net.layer_sizes[k]):
            for j in range(net.layer_sizes[k+1]):
                edges.append((k, i, j))
                edge_scores.append(abs(ratio_weighted[k][j, i]))
    edges = np.array(edges)
    edge_scores = np.array(edge_scores)

    # Sort nodes by descending order of edge score
    sort_inds = np.argsort(edge_scores)[::-1]
    edges = edges[sort_inds]

    # Prune up to `num_edges` by prune_factor
    pruned_net = SimpleNet()
    pruned_net.load_state_dict(net.state_dict())
    for k, i, j in edges[:num_edges]:
        pruned_net = prune_edge(pruned_net, k, i, j, prune_factor=prune_factor,
                                return_copy=False)

    return pruned_net


def prune(net, z_info_flows, y_info_flows, prune_factor, params):
    """
    Prune ANN based on the method and metric specified in params
    """
    accbias = (params.prune_metric == 'accbias')

    if params.prune_method == 'node':
        return prune_nodes_biasacc(net, z_info_flows, y_info_flows,
                                   num_nodes=params.num_to_prune,
                                   prune_factor=prune_factor, accbias=accbias)
    elif params.prune_method == 'edge':
        return prune_edges_proportional(net, z_info_flows, y_info_flows,
                                        num_edges=params.num_to_prune,
                                        prune_factor=prune_factor, accbias=accbias)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    params = init_params()
    #params.force_regenerate = True  # Only relevant for tinyscm
    #params.force_retrain = True

    #data = init_data(params, dataset='tinyscm')
    data = init_data(params, dataset='adult')
    print(params.num_data, params.num_train)

    if params.force_retrain or params.annfile is None:
        net = train_ann(data, params, test=False, savefile=params.annfile)
    else:
        net = SimpleNet()
        net.load_state_dict(torch.load(params.annfile))
