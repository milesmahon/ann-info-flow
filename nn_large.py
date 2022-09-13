#!/usr/bin/env python3

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

# The neural network class definitions need to be in a separate module because
# of the way joblib works:
# https://stackoverflow.com/questions/49621169/joblib-load-main-attributeerror


class SmallNet(nn.Module):
    """
    A simple neural network with no convolutional elements. It uses a
    fully-connected architecture with a fixed number of layers and neurons
    in each layer.
    """

    def __init__(self):
        """Initializes a neural network with one hidden layer."""
        super(SmallNet, self).__init__()
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
        x2 = self.fc2(x1)
        self.activations = [x0, x1, x2]
        return x2

    def get_weights(self):
        return [getattr(self, 'fc%d' % i).weight.data.numpy()
                for i in range(1, self.num_layers)]

    def set_weights(self, weights):
        for i, w in enumerate(weights):
            getattr(self, 'fc%d' % (i + 1)).weight.data.numpy()[:, :] = w


class LargeNet(nn.Module):
    """
    A simple neural network with no convolutional elements. It uses a
    fully-connected architecture with a fixed number of layers and neurons
    in each layer.
    """

    def __init__(self):
        """Initializes a neural network with one hidden layer."""
        super(LargeNet, self).__init__()
        self.workclass = nn.Embedding(9,1)
        self.occupation = nn.Embedding(15,1)
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 2)  # 1-hot encoding of 0/1
        self.leaky_relu = nn.LeakyReLU()
        self.num_layers = 4  # Includes input and output layer
        self.layer_sizes = [5, 3, 3, 2]

    def forward(self, x):
        x0 = x.detach().clone()
        x0[:,0] = self.occupation(x0[:,0].type(torch.LongTensor)).squeeze(1)
        x0[:,1] = self.workclass(x0[:,1].type(torch.LongTensor)).squeeze(1) 
        x1 = self.leaky_relu(self.fc1(x0))
        x2 = self.leaky_relu(self.fc2(x1))
        x3 = self.fc3(x2)
        self.activations = [x0, x1, x2, x3]
        return x3

    def get_weights(self):
        return [getattr(self, 'fc%d' % i).weight.data.numpy()
                for i in range(1, self.num_layers)]

    def set_weights(self, weights):
        for i, w in enumerate(weights):
            getattr(self, 'fc%d' % (i + 1)).weight.data.numpy()[:, :] = w


