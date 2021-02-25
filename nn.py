#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        #self.fc2 = nn.Linear(3, 1)  # Thresholded encoding of 0/1
        self.fc2 = nn.Linear(3, 2)  # 1-hot encoding of 0/1
        self.num_layers = 3  # Includes input and output layer
        #self.layer_sizes = [3, 3, 1]
        self.layer_sizes = [3, 3, 2]

    def forward(self, x):
        x0 = x.view(-1, 3)
        leaky_relu = nn.LeakyReLU()
        #x1 = torch.sigmoid(self.fc1(x0))
        #x2 = torch.sigmoid(self.fc2(x1))
        x1 = leaky_relu(self.fc1(x0))
        x2 = leaky_relu(self.fc2(x1))
        self.activations = [x0, x1, x2]
        return x2

    def get_weights(self):
        return [getattr(self, 'fc%d' % i).weight.data.numpy()
                for i in range(1, self.num_layers)]

    def set_weights(self, weights):
        for i, w in enumerate(weights):
            getattr(self, 'fc%d' % (i + 1)).weight.data.numpy()[:, :] = w


def train_ann(data, params, test=False, random_seed=None):
    # TODO: Implement early stopping and weight initialization
    if random_seed is None:
        torch.manual_seed(13)
    else:
        torch.manual_seed(random_seed)

    X, Y = data.data[:2]
    X = torch.from_numpy(np.array(X)).float()
    #Y = torch.from_numpy(np.array(Y).reshape((-1, 1))).float()
    Y = torch.from_numpy(np.array(Y).reshape((-1, 1))).long()

    # Separate training and testing data
    X_train = X[:params.num_train]
    Y_train = Y[:params.num_train]

    # Training parameters
    num_epochs = params.num_epochs[data.dataset]
    minibatch_size = params.minibatch_size[data.dataset]
    learning_rate = params.learning_rate[data.dataset]
    momentum = params.momentum[data.dataset]
    print_every = (params.num_train // minibatch_size
                   // params.print_every_factor[data.dataset])

    # Set up neural network, loss function and optimizer
    net = SimpleNet()
    net.train()
    criterion = params.criterion[data.dataset]
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i in range(params.num_train // minibatch_size):
            # Get minibatch from training data
            x = X_train[i*minibatch_size : (i+1)*minibatch_size, :]
            y = Y_train[i*minibatch_size : (i+1)*minibatch_size, :]

            # zero the parameter gradients
            optimizer.zero_grad()

            # Compute loss on x, backprop, and take a gradient step
            yhat = net(x)
            loss = criterion(torch.squeeze(yhat), torch.squeeze(y))
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % print_every == 0:
                print('[%d, %3d] loss: %.3f' %
                      (epoch + 1, i, running_loss / minibatch_size))
                running_loss = 0.0

    print('Finished Training')

    if test:
        net.eval()
        num_test = params.num_data - params.num_train
        X_test = X[params.num_train:]
        Y_test = Y[params.num_train:]

        with torch.no_grad():
            Yhat = net(X_test)
            predictions = (Yhat > 0.5)
            correct = (predictions == Y_test).sum().item()

        print('Accuracy: %d%%' % (100 * correct / num_test))

    return net
