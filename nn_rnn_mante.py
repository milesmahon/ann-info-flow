import os.path
import sys
import time

import requests
import torch
import torch.nn as nn
import numpy as np
from numpy import nonzero

from analyze_info_flow import analyze_info_flow_rnn
from datasets.MotionColorDataset import MotionColorDataset
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 50000  # 100,000 takes around 2 minutes w/ 1 layer hidden size 4,
learning_rate = 0.0001
hidden_size = 4
num_layers = 1
batch_size = 1000
input_size = 3  # (motion (float), color (float), context (bool/int))
output_size = 2  # (-1, 1) one-hot encode
debug = False

# Dataset params
# train_dataset = MotionColorDataset(100, 10)
# test_dataset = MotionColorDataset(100, 10)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity="tanh")
        self.fc1 = nn.Linear(hidden_size, output_size)
        # self.fc2 = nn.Linear(16, output_size)
        self.activations = []

    def forward(self, x, hidden):
        x1, hidden_i = self.rnn(x, hidden)
        x2 = self.fc1(hidden_i)  # was x1 instead of hidden
        # x3 = self.fc2(x2)
        # hidden state is equivalent to activation of RNN layer, no need to pass hidden_i to activations
        self.activations = [x1, x2]
        return x2, hidden_i

    def init_hidden(self, batch_size=None):
        if batch_size == None:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float64).to(device)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float64).to(device)

    def get_weights(self):
        weights = []
        # all_weights returns a 4-component list:
        # ih weights, hh weights, ih bias, hh bias
        for layer in self.rnn.all_weights[0][0:2]:
            weights.append(np.array(layer.data))
        weights.append(getattr(self, 'fc1').weight.data.numpy())  # TODO MM update if multiple layers
        # weights.append([getattr(self, 'fc%d' % i).weight.data.numpy() for i in range(1, 3)])
        return weights


def translate_to_cel(label):
    return nonzero(label == 1.0)[0]


FILE = "model.pth"


def train_new_rnn():
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print('training model')
    time_start = time.perf_counter()
    model.train()
    for i in range(5):  # train on 5 sets of batch_size
        mc_dataset = MotionColorDataset(batch_size, 10)
        X, _, _, true_labels, _ = mc_dataset.get_xyz(batch_size, context_time="retro", vary_acc=True)
        X = np.array(X)
        Y = np.array(true_labels)
        for epoch in range(num_epochs):
            hidden = model.init_hidden()
            output, hidden = model(torch.from_numpy(X).float(), hidden.float())
            optimizer.zero_grad()
            loss = criterion(torch.squeeze(output), torch.from_numpy(Y))
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{n_total_steps}], Loss: {loss.item() * 1000:.6f}')
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item() * 1000:.6f}')
                print(f'Time elapsed: {time.perf_counter() - time_start:0.2f} seconds')
    model.eval()
    return model


def test_rnn(net):
    mc_dataset = MotionColorDataset(1000, 10)  # TODO pass dataset from training
    X, _, _, true_labels, _ = mc_dataset.get_xyz(1000, context_time='retro', vary_acc=True)
    X_test = np.array(X)  # input
    U_test = np.array(true_labels)  # true label
    with torch.no_grad():
        num_test = 1000
        # Compute and print test accuracy
        net.eval()
        correct = 0
        # for i in range(num_test):
        hidden = net.init_hidden()
        output, hidden = net(torch.from_numpy(X_test).float(), hidden.float())
        # Extract intermediate activations from the network
        # Xint = [actvn.numpy() for actvn in net.activations]

        Yhat = np.squeeze(output.numpy())
        #predictions = (Yhat > 0.5)  # Logic for single output node encoding 0/1 using a sigmoid
        predictions = (Yhat[:, 0] < Yhat[:, 1]).astype(int)  # Logic for 1-hot encoding of 0/1 at output node
        correct = (predictions == U_test).sum()
        accuracy = correct / num_test

        print('Accuracy: %.5g%%' % (100 * accuracy))
    return accuracy

model = RNN(input_size, hidden_size, num_layers, output_size, batch_size)
model.load_state_dict(torch.load('rnn-tests-22-11-15-4node/model8.pth'))
acc_before_pruning = test_rnn(model)
print(acc_before_pruning)
print(model.rnn.all_weights)
z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy = \
            analyze_info_flow_rnn(model, 'corr', model_name='model-pre-prune')

with torch.no_grad():
    model.rnn.all_weights[0][1][3][1] = 0  # prune the self hh weight for node bottom node from itself

print(model.rnn.all_weights)
z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy = \
            analyze_info_flow_rnn(model, 'corr', model_name='model-post-prune-1')

# if os.path.isfile(FILE):
#     model.load_state_dict(torch.load(FILE))
# else:
url = ''
# for i in range(10):
#     model = train_new_rnn()
#     torch.save(model.state_dict(), 'model' + str(i) + '.pth')
#     acc = test_rnn(model)
#     myobj = {'text': str(acc)}
#     x = requests.post(url, json=myobj)
#     print(x.text)

# for i in range(10):
#     model = RNN(input_size, hidden_size, num_layers, output_size, batch_size)
#     model.load_state_dict(torch.load('model'+str(i)+'.pth'))
#     print("testing " + str(i))
#     stdout = sys.stdout
#     with open('rnn-tests-22-11-15-2node/model'+str(i)+'.txt', 'w') as f:
#         sys.stdout = f
#         os.makedirs('rnn-tests-22-11-15-2node/model'+str(i))
#         z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy = \
#             analyze_info_flow_rnn(model, 'corr', model_name='model'+str(i))
#         sys.stdout = stdout
#     requests.post(url, json={'text': 'analyzed '+str(i)})


# z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy = \
#     analyze_info_flow_rnn(model, 'corr')
    # analyze_info_flow_rnn(model, 'linear-svm')  # use corr for estimate via correlation
