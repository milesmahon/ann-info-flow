#!/usr/bin/env python3

from __future__ import print_function, division

import argparse
import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from param_utils import init_params
from data_utils import init_data, print_data_stats


class FullNet(nn.Module):
    """
    A simple neural network with no convolutional elements. It uses a
    fully-connected architecture with a fixed number of layers and neurons
    in each layer.
    """

    def __init__(self):
        """Initializes a neural network with one hidden layer."""
        super(FullNet, self).__init__()
        self.conv = nn.Conv2d(1,6,28,stride=1)
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 6)
        self.fc4 = nn.Linear(6, 6)
        self.fc5 = nn.Linear(6, 2)
        self.relu = nn.LeakyReLU()
        self.num_layers = 7  # Includes input and output layer
        self.layer_sizes = [784, 6, 6, 6, 6, 6, 2]

    def forward(self, x):
        x0 = x.unsqueeze(1)#.reshape(x.shape[0],784)
        x1 = self.conv(x0)
        x2 = self.relu(self.fc1(x1.reshape(x.shape[0],6)))
        x3 = self.relu(self.fc2(x2))
        x4 = self.relu(self.fc3(x3))
        x5 = self.relu(self.fc4(x4))
        x6 = self.fc5(x5)

        self.activations = [x0, x1, x2, x3, x4, x5, x6]
        return x6,x1.reshape(x.shape[0],6)

    def get_weights(self):
        return [getattr(self, 'fc%d' % i).weight.data.numpy()
                for i in range(1, self.num_layers)]

    def set_weights(self, weights):
        for i, w in enumerate(weights):
            getattr(self, 'fc%d' % (i + 1)).weight.data.numpy()[:, :] = w


class SimpleNet(nn.Module):
    """
    A simple neural network with no convolutional elements. It uses a
    fully-connected architecture with a fixed number of layers and neurons
    in each layer.
    """

    def __init__(self):
        """Initializes a neural network with one hidden layer."""
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 6)
        self.fc4 = nn.Linear(6, 6)
        self.fc5 = nn.Linear(6, 2)
        self.relu = nn.LeakyReLU()
        self.num_layers = 6  # Includes input and output layer
        self.layer_sizes = [6, 6, 6, 6, 6, 2]

    def forward(self, x):
        x1 = x
        x2 = self.relu(self.fc1(x1))
        x3 = self.relu(self.fc2(x2))
        x4 = self.relu(self.fc3(x3))
        x5 = self.relu(self.fc4(x4))
        x6 = self.fc5(x5)

        self.activations = [x1, x2, x3, x4, x5, x6]
        return x6

    def get_weights(self):
        return [getattr(self, 'fc%d' % i).weight.data.numpy()
                for i in range(1, self.num_layers)]

    def set_weights(self, weights):
        for i, w in enumerate(weights):
            getattr(self, 'fc%d' % (i + 1)).weight.data.numpy()[:, :] = w


def train_ann(run, data, params, test=False, random_seed=None):
    # TODO: Implement early stopping and weight initialization
    if random_seed is None:
        torch.manual_seed(13)
    else:
        torch.manual_seed(random_seed)

    X, Y = data.data[:2]
    X = torch.from_numpy(np.stack(X).squeeze(1)).float()
    Y = torch.from_numpy(Y.reshape((-1, 1))).long()

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
    net = FullNet()
    net.train()
    criterion = params.criterion[data.dataset]
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(num_epochs):
        running_loss = 0.0
        avg_loss = 0.0

        perm_inds = torch.randperm(X_train.shape[0])
        X_train = X_train[perm_inds, :]
        Y_train = Y_train[perm_inds, :]

        for i in range(params.num_train // minibatch_size):
            # Get minibatch from training data
            x = X_train[i*minibatch_size : (i+1)*minibatch_size, :]
            y = Y_train[i*minibatch_size : (i+1)*minibatch_size, :]

            # zero the parameter gradients
            optimizer.zero_grad()

            # Compute loss on x, backprop, and take a gradient step
            yhat,_ = net(x)
            loss = criterion(torch.squeeze(yhat), torch.squeeze(y))
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            avg_loss += loss.item()
            if i % print_every == print_every - 1:
                print('[%d, %3d] loss: %.3f' %
                      (epoch + 1, i, running_loss / (minibatch_size * print_every)))
                running_loss = 0.0

        avg_loss /= X_train.shape[0]
        scheduler.step(avg_loss)

    print('Finished Training')

    if test:
        net.eval()
        num_test = params.num_data - params.num_train
        X_test = X[params.num_train:]
        Y_test = Y[params.num_train:]

        with torch.no_grad():
            Yhat,_ = net(X_test)
            #print(X1.shape)
            predictions = torch.argmax(Yhat,dim=1)
            correct = (predictions == Y_test.squeeze(1)).sum().item()

        print('Accuracy: %d%%' % (100 * correct / num_test))

    pred,procX = net(X)

    filters = net.state_dict()["conv.weight"].reshape(28,28,6)
    fig, axs = plt.subplots(1,6)
    axs[0].imshow(filters[:,:,0])
    axs[1].imshow(filters[:,:,1])
    axs[2].imshow(filters[:,:,2])
    axs[3].imshow(filters[:,:,3])
    axs[4].imshow(filters[:,:,4])
    axs[5].imshow(filters[:,:,5])
    fig.savefig("./results-mnist/filters/run-" + str(run) + ".png", dpi=fig.dpi)

    del net.conv

    trunc_net = SimpleNet()

    trunc_net.load_state_dict(net.state_dict())
    print(procX.shape)
    print(pred.shape)

    trunc_pred = trunc_net(procX).detach().numpy()
    pred = pred.detach().numpy()
    print(trunc_pred.shape)

    mismatch_cnt = 0
    if np.allclose(pred,trunc_pred) == False:
        print("FullNet and SimpleNet Predictions DO NOT MATCH.")
        mismatch_cnt += 1

    else:
        print("FullNet and SimpleNet Predictions MATCH.")

    proc_data = [procX.detach().numpy(), Y.detach().numpy(), data.data[2]]

    return trunc_net, proc_data, mismatch_cnt


if __name__ == '__main__':
    params = init_params()

    parser = argparse.ArgumentParser(description='Train ANN on a dataset')
    parser.add_argument('-d', '--dataset', choices=params.datasets, help='Dataset to use for analysis')
    parser.add_argument('--runs', type=int, help='Number of times to run the analysis.')
    args = parser.parse_args()
    # ANN training appears fast enough to not need parallelization;
    # implement GPU if speed becomes an issue.

    print('\n------------------------------------------------')
    print(args)
    print('------------------------------------------------\n')

    # Override params specified in param_utils, if given in CLI
    if args.dataset:
        # Reinitialize params if the dataset is given
        params = init_params(dataset=args.dataset)
    if args.runs:
        params.num_runs = args.runs

    # For now, keep data fixed across runs
    data = init_data(params)
    print(params.num_data, params.num_train)
    #print_data_stats(data, params) # Check data statistics

    # Train nets and save to file
    nets = []
    procData=[]
    for run in range(params.num_runs):
        print('------------------')
        print('Run %d' % run)
        net, proc_data, err_cnt = train_ann(run, data, params, test=True, random_seed=(1000+run))
        nets.append(net)
        procData.append(proc_data)
    print("Bad Runs: ", err_cnt)
    joblib.dump(nets, params.annfile, compress=3)
    joblib.dump(procData, params.procdatafile, compress=3)
