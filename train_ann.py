#!/usr/bin/env python3

from __future__ import print_function, division

import argparse
import joblib
import numpy as np

import torch
import torch.optim as optim

from nn_large import SmallNet, LargeNet
from param_utils import init_params
from data_utils_large import init_data, print_data_stats


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
    if data.dataset in ['tinyscm', 'adult-small']:
        net = SmallNet()
    elif data.dataset == 'adult-large':
        net = LargeNet()
    else:
        raise ValueError('Unrecognized dataset %s' % data.dataset)

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
            yhat = net(x)
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
            Yhat = net(X_test)
            predictions = torch.argmax(Yhat, dim=1)
            correct = (predictions == Y_test.squeeze(1)).sum().item()

        print('Accuracy: %d%%' % (100 * correct / num_test))

    return net


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
    print_data_stats(data, params) # Check data statistics

    # Train nets and save to file
    nets = []
    for run in range(params.num_runs):
        print('------------------')
        print('Run %d' % run)
        net = train_ann(data, params, test=True, random_seed=(1000+run))
        nets.append(net)
    joblib.dump(nets, params.annfile, compress=3)
