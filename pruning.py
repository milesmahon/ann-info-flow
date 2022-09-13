#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import copy
from info_measures import weight_info_flows


def edge_list(net, scores=None):
    """
    Make an array of all edges in a network. Optionally, make an additional
    array of all edge scores provided in scores, prepared in the same order.
    """

    # Vectorize weighted ratios to create edge scores
    edges = []
    if scores: edge_scores = []
    for k in range(net.num_layers - 1):
        for i in range(net.layer_sizes[k]):
            for j in range(net.layer_sizes[k+1]):
                edges.append((k, i, j))
                if scores: edge_scores.append(scores[k][j, i])
    edges = np.array(edges)
    if scores: edge_scores = np.array(edge_scores)

    if scores:
        return edges, edge_scores
    else:
        return edges


def prune_edge(net, layer, i, j, prune_factor=0, return_copy=True):
    """
    Prunes an edge from neuron `i` to neuron `j` at a given `layer` in `net`
    by `prune_factor` and returns a copy (if `return_copy` is True).
    """

    # Create a copy if required
    if return_copy:
        pruned_net = type(net)()
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
        pruned_net = type(net)()
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


def prune_edges_random(net, num_edges=1, prune_factor=0):
    """
    Prune `num_edges` randomly from the network.
    """

    edges = edge_list(net)
    # Choose edges at random to prune
    rng = np.random.default_rng()
    edge_ids_to_prune = rng.choice(edges.shape[0], num_edges, replace=False)

    pruned_net = type(net)()
    pruned_net.load_state_dict(net.state_dict())
    for k, i, j in edges[edge_ids_to_prune]:
        pruned_net = prune_edge(pruned_net, k, i, j, prune_factor=prune_factor,
                                return_copy=False)

    return pruned_net


def prune_nodes_random(net, num_nodes=1, prune_factor=0):
    """
    Prune `num_nodes` randomly from the network.
    """

    node_list = []
    for k in range(net.num_layers - 1):
        node_list.extend([[k, i] for i in range(net.layer_sizes[k])])
    node_list = np.array(node_list)

    # Choose nodes at random to prune
    rng = np.random.default_rng()
    node_ids_to_prune = rng.choice(node_list.shape[0], num_nodes, replace=False)

    pruned_net = type(net)()
    pruned_net.load_state_dict(net.state_dict())
    for k, i in node_list[node_ids_to_prune]:
        pruned_net = prune_node(pruned_net, k, i, prune_factor=prune_factor,
                                return_copy=False)

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
    pruned_net = type(net)()
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

    edges, edge_scores = edge_list(net, ratio_weighted)

    # Sort nodes by descending order of edge score
    sort_inds = np.argsort(abs(edge_scores))[::-1]
    edges = edges[sort_inds]

    # Prune up to `num_edges` by prune_factor
    pruned_net = type(net)()
    pruned_net.load_state_dict(net.state_dict())
    for k, i, j in edges[:num_edges]:
        pruned_net = prune_edge(pruned_net, k, i, j, prune_factor=prune_factor,
                                return_copy=False)

    return pruned_net


def prune_path(net, z_info_flows, y_info_flows, num_paths=1, prune_factor=0, accbias=False):
    """
    Prune edges by identifying the "information path" carrying the largest flow
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

    edges, flows = edge_list(net, ratio_weighted)
    flows = abs(flows)
    edges_tuple = [tuple(e) for e in edges]  # Convert into a list of tuples
    flows_dict = dict(zip(edges_tuple, flows))

    # Set of all paths, identified by (node_in_layer_0, node_in_layer_1, ...)
    a = np.array(np.meshgrid(*[range(k) for k in net.layer_sizes], indexing='ij'))  # All points in a 3D grid within the given ranges
    a = np.rollaxis(a, 0, net.num_layers+1)                                         # Make the 0th axis into the last axis
    path_nodes = a.reshape((-1, net.num_layers))                                    # Now you can safely reshape while preserving order

    # Set of all paths, identified by edges
    path_edges = []
    for path in path_nodes:
        path_edge = []
        for i in range(len(path) - 1):
            path_edge.append((i, path[i], path[i+1]))
        path_edges.append(path_edge)

    # Find min of weights of edges in each path
    path_weights = [min(flows_dict[edge] for edge in path) for path in path_edges]
    path_weights = np.array(path_weights)
    # Find indices to sort path weights in descending order
    sort_inds = np.argsort(path_weights)[::-1]

    if prune_factor < 0.01:
        print(path_weights[sort_inds])

    # Prune up to `num_paths` by prune_factor
    # NOTE: It really doesn't make sense to prune multiple paths right now. If
    # two paths overlap, then the common edges will suffer squared pruning
    pruned_net = type(net)()
    pruned_net.load_state_dict(net.state_dict())
    for ind in sort_inds[:num_paths]:
        edges = path_edges[ind]          # Pick all edges in the path
        #edges = [path_edges[ind][-1],]  # Pick only the edge at the final layer: works okay, but doesn't do enough
        for k, i, j in edges:
            pruned_net = prune_edge(pruned_net, k, i, j, prune_factor=prune_factor,
                                    return_copy=False)

    return pruned_net


def prune(net, z_info_flows, y_info_flows, prune_factor, params):
    """
    Prune ANN based on the method and metric specified in params
    """
    accbias = (params.prune_metric == 'accbias')

    if params.prune_metric == 'random':
        if params.prune_method == 'node':
            return prune_nodes_random(net, num_nodes=params.num_to_prune,
                                      prune_factor=prune_factor)
        elif params.prune_method == 'edge':
            return prune_edges_random(net, num_edges=params.num_to_prune,
                                      prune_factor=prune_factor)
        else:
            raise NotImplementedError()

    if params.prune_method == 'node':
        return prune_nodes_biasacc(net, z_info_flows, y_info_flows,
                                   num_nodes=params.num_to_prune,
                                   prune_factor=prune_factor, accbias=accbias)
    elif params.prune_method == 'edge':
        return prune_edges_proportional(net, z_info_flows, y_info_flows,
                                        num_edges=params.num_to_prune,
                                        prune_factor=prune_factor, accbias=accbias)
    elif params.prune_method == 'path':
        return prune_path(net, z_info_flows, y_info_flows,
                          num_paths=params.num_to_prune,
                          prune_factor=prune_factor, accbias=accbias)
    else:
        raise NotImplementedError()
