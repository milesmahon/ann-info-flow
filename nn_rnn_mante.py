import time
import pickle

import torch
import torch.nn as nn
import numpy as np
from numpy import nonzero
from torch.optim.lr_scheduler import MultiStepLR

from analyze_info_flow import test_rnn
from datasets.MotionColorDataset import MotionColorDataset
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10000  # 100,000 takes around 2 minutes w/ 1 layer hidden size 4, 1k batch size
learning_rate = 0.01  # .0001
hidden_size = 4
num_layers = 1
batch_size = 10000
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
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity="tanh")
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.activations = []

    def forward(self, x, hidden):
        x1, hidden_i = self.rnn(x, hidden)
        x2 = self.fc1(hidden_i)
        # hidden state is equivalent to activation of RNN layer, no need to pass hidden_i to activations
        self.activations = [x1, x2]
        return x2, hidden_i

    def init_hidden(self, batch_size=None):
        if batch_size is None:
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
    preds = []
    accs = []
    opt_accs = []
    not_adhere_sample_means_list = []
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer,
                        milestones=[1000, 3000, 5000],  # List of epoch indices
                        gamma =0.1)
    print('training model')
    time_start = time.perf_counter()
    model.train()
    mc_dataset = MotionColorDataset(batch_size, 10)
    X, _, _, true_labels, _ = mc_dataset.get_xyz(batch_size, context_time="pro", vary_acc=True)
    X = np.array(X)
    Y = np.array(true_labels)
    for epoch in range(num_epochs):
        hidden = model.init_hidden()
        output, hidden = model(torch.from_numpy(X).float(), hidden.float())
        optimizer.zero_grad()
        loss = criterion(torch.squeeze(output), torch.from_numpy(Y))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item() * 1000:.6f}')
            print(f'Time elapsed: {time.perf_counter() - time_start:0.2f} seconds')
            model.eval()
            pred_match_true, acc, opt_acc, not_adhere_sample_means = test_rnn(model)
            preds.append(pred_match_true)
            accs.append(acc)
            opt_accs.append(opt_acc)
            not_adhere_sample_means_list.append(sum(not_adhere_sample_means)/len(not_adhere_sample_means))
            print(pred_match_true/10000)
            print(acc)
            print(scheduler.get_last_lr())
            model.train()
    model.eval()
    print(preds)
    print(accs)
    return model, preds, accs, opt_accs, not_adhere_sample_means_list


model, preds, accs, opt_accs, not_adhere_sample_means_list = train_new_rnn()
print(preds, accs)

open_file = open("preds.pkl", "wb")
pickle.dump(preds, open_file)
open_file.close()

open_file = open("accs.pkl", "wb")
pickle.dump(accs, open_file)
open_file.close()

open_file = open("preds.pkl", "rb")
preds = pickle.load(open_file)
open_file.close()

open_file = open("accs.pkl", "rb")
accs = pickle.load(open_file)
open_file.close()

plt.plot([x/10000 for x in preds])
plt.xlabel("Epoch (thousands)")
plt.ylabel("Predictions matching sign of sample mean (percent)")
plt.title("Adherence to sample mean over training (10k epochs)")
plt.show()

plt.plot(accs)
plt.plot(opt_accs)
plt.legend(['accuracy achieved', 'optimal accuracy'], loc='upper left')
plt.xlabel("Epoch (thousands)")
plt.ylabel("Accuracy (percent)")
plt.title("Accuracy over training (10k epochs)")
plt.show()
print("max preds: ", max(preds))
print("max accs: ", max(accs))

plt.plot(not_adhere_sample_means_list)
plt.xlabel("Epoch (thousands)")
plt.ylabel("Average sample mean when not adhering")
plt.title("Average sample mean in non-adhere cases over training (10k epochs)")
plt.show()
