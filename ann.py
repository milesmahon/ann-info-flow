#!/usr/bin/env python3

from __future__ import print_function, division

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from data_utils import generate_data
from info_measures import mutual_info_bin, compute_all_flows, weight_info_flows
from utils import powerset, print_mis, print_edge_data, print_node_data


class SimpleNet(nn.Module):
    """
    A simple neural network with no convolutional elements. It uses a
    fully-connected architecture with a fixed number of layers and neurons
    in each layer.
    """

    def __init__(self):
        """Initializes a neural network with one hidden layer."""
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x0 = x.view(-1, 3)
        x1 = torch.sigmoid(self.fc1(x0))
        x2 = torch.sigmoid(self.fc2(x1))
        self.activations = [x0, x1, x2]
        return x2


def train_ann(data, num_data, num_train, test=False, savefile=None):
    # TODO: Implement early stopping

    X, Y = data[2:]
    X = torch.from_numpy(np.array(X).T).float()
    Y = torch.from_numpy(np.array(Y).reshape((-1, 1))).float()

    # Separate training and testing data
    X_train = X[:num_train]
    Y_train = Y[:num_train]

    # Training parameters
    num_epochs = 10
    minibatch_size = 10  # Should be a factor of num_train
    learning_rate = 0.1
    momentum = 0.9
    print_every = num_train // minibatch_size // 10

    # Set up neural network, loss function and optimizer
    net = SimpleNet()
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i in range(num_train // minibatch_size):
            # Get minibatch from training data
            x = X_train[i*minibatch_size : (i+1)*minibatch_size, :]
            y = Y_train[i*minibatch_size : (i+1)*minibatch_size, :]

            # zero the parameter gradients
            optimizer.zero_grad()

            # Compute loss on x, backprop, and take a gradient step
            yhat = net(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % print_every == 0:
                print('[%d, %3d] loss: %.3f' %
                      (epoch + 1, i, running_loss / minibatch_size))
                running_loss = 0.0

    print('Finished Training')

    if savefile:
        torch.save(net.state_dict(), savefile)

    if test:
        num_test = num_data - num_train
        X_test = X[num_train:]
        Y_test = Y[num_train:]

        with torch.no_grad():
            Yhat = net(X_test)
            predictions = (Yhat > 0.5)
            correct = (predictions == Y_test).sum().item()

        print('Accuracy: %d%%' % (100 * correct / num_test))

    return net


def corrcoef(x, y):
    """
    Correlate an (n, m) array and an (n, r) array to get an
    (m, r) cross-correlation matrix.
    """
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)

    cov = np.einsum('nm,nr->mr', x, y) / x.shape[0]
    sigma_x = np.std(x, axis=0)
    sigma_y = np.std(y, axis=0)

    return cov / (sigma_x[:, None] * sigma_y)


def analyze_info_flow(net, data, num_data, num_train):
    Z, U, X, Y = data
    Z = np.array(Z)
    U = np.array(U).T
    X = np.array(X).T
    Y = np.array(Y)

    Z_test = Z[num_train:]
    U_test = U[num_train:]
    Y_test = Y[num_train:]

    # PyTorch stuff
    with torch.no_grad():
        X_ = torch.from_numpy(X).float()
        Y_ = torch.from_numpy(np.array(Y).reshape((-1, 1))).float()

        num_test = num_data - num_train
        X_test = X_[num_train:]

        # Compute and print test accuracy
        Yhat = net(X_test).numpy().flatten()
        predictions = (Yhat > 0.5)
        correct = (predictions == Y_test).sum()
        print('Accuracy: %d%%\n' % (100 * correct / num_test))

        # Extract intermediate activations from the network
        Xint = [actvn.numpy() for actvn in net.activations]

    num_layers = len(Xint)
    layer_sizes = [(Xint_.shape[1] if Xint_.ndim > 1 else 1) for Xint_ in Xint]

    z_corr = [corrcoef(Z_test[:, None], Xint[i]) for i in range(num_layers)]
    print('Correlations with Z:')
    for corr in z_corr: print(corr)
    print()

    print('Correlations with Y:')
    y_corr = [corrcoef(Y_test[:, None], Xint[i]) for i in range(num_layers)]
    for corr in y_corr: print(corr)
    print()

    print('Weights')
    weights = [getattr(net, 'fc%d' % i).weight.data.numpy()
               for i in range(1, num_layers)]
    print_edge_data(weights, layer_sizes)
    print()

    header = 'Layer\tX1\tX2\tX3\tX12\tX13\tX23\tX123'

    print('Computing bias flows...')
    print('Accuracies:')
    print(header)
    z_mis = [None,] * num_layers
    for i in range(num_layers):
        print(i, end='', flush=True)
        z_mis[i] = {(): 0}
        if Xint[i].ndim == 1:
            Xint[i] = Xint[i].reshape((-1, 1))
        for js in powerset(range(layer_sizes[i]), start=1):
            z_mi, z_acc = mutual_info_bin(Z_test, Xint[i][:, js], Hx=1, return_acc=True)
            z_mis[i][js] = z_mi
            print('\t%.4f' % z_acc, end='', flush=True)
        print()
    print('Mutual informations:')
    print_mis(z_mis, layer_sizes, header)
    print('Information flows:')
    z_info_flows = compute_all_flows(z_mis, layer_sizes)
    print_node_data(z_info_flows, layer_sizes)
    print('Weighted information flows:')
    z_info_flows_weighted = weight_info_flows(z_info_flows, weights)
    print_edge_data(z_info_flows_weighted, layer_sizes)
    print()

    print('Computing accuracy flows...')
    print('Accuracies:')
    print(header)
    y_mis = [None,] * num_layers
    for i in range(num_layers):
        print(i, end='', flush=True)
        y_mis[i] = {(): 0}
        if Xint[i].ndim == 1:
            Xint[i] = Xint[i].reshape((-1, 1))
        for js in powerset(range(layer_sizes[i]), start=1):
            y_mi, y_acc = mutual_info_bin(Y_test, Xint[i][:, js], Hx=1, return_acc=True)
            y_mis[i][js] = y_mi
            print('\t%.4f' % y_acc, end='', flush=True)
        print()
    print('Mutual informations:')
    print_mis(y_mis, layer_sizes, header)
    print('Information flows:')
    y_info_flows = compute_all_flows(y_mis, layer_sizes)
    print_node_data(y_info_flows, layer_sizes)
    print('Weighted information flows:')
    y_info_flows_weighted = weight_info_flows(y_info_flows, weights)
    print_edge_data(y_info_flows_weighted, layer_sizes)
    print()

    #plt.figure()
    #mask = (Yhat > 0.5)
    #plt.plot(U_test[mask, 0], U_test[mask, 1], 'C0o', alpha=0.3)
    #plt.plot(U_test[~mask, 0], U_test[~mask, 1], 'C1o', alpha=0.3)

    #plt.figure()
    #mask = (Z_test > 0.5)
    #plt.plot(U_test[mask, 0], Xint[2][mask, 0], 'C0o', alpha=0.3)
    #plt.plot(U_test[~mask, 0], Xint[2][~mask, 0], 'C1o', alpha=0.3)

    #plt.show()


if __name__ == '__main__':
    num_data = 2000
    #force_regenerate = False
    force_regenerate = True
    datafile = 'data-%d.pkl' % num_data
    # Load data if it exists; generate and save data otherwise
    if force_regenerate or not os.path.isfile(datafile):
        data = generate_data(num_data)
        print('Data generation complete')
        joblib.dump(data, datafile, compress=3)
    else:
        data = joblib.load(datafile)

    num_train = 1000
    annfile = 'nn-state.sav'
    net = train_ann(data, num_data, num_train, test=False, savefile=annfile)
    #net = SimpleNet()
    #net.load_state_dict(torch.load(annfile))

    analyze_info_flow(net, data, num_data, num_train)
