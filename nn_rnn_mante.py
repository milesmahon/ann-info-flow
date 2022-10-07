import time

import torch
import torch.nn as nn
import numpy as np
from numpy import nonzero

from analyze_info_flow import analyze_info_flow_rnn
from datasets.MotionColorDataset import MotionColorDataset
from plot_utils import plot_ann
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 5
learning_rate = 0.0001
hidden_size = 3
num_layers = 1  # TODO MM try multiple layers
batch_size = 100
input_size = 3  # (motion (float), color (float), context (bool/int))
output_size = 2  # (-1, 0, 1) one-hot encoded
debug = False

# Dataset params
train_dataset = MotionColorDataset(100, 10)
test_dataset = MotionColorDataset(100, 10)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        # self.fc2 = nn.Linear(16, output_size)
        self.activations = []

    def forward(self, x, hidden):
        x1, hidden_i = self.rnn(x, hidden)
        x2 = self.fc1(hidden_i)  # was x1 instead of hidden
        # x3 = self.fc2(x2)
        # hidden state is equivalent to activation of RNN layer, no need to pass hidden_i to activations
        self.activations = [x1, x2]  # TODO MM will miss initial hidden state this way
                                    # TODO MM include output (x2)?
        return x2, hidden_i

    def init_hidden(self, batch_size=None):
        if batch_size == None:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def get_weights(self):
        # TODO MM verify this output makes sense
        weights = []
        # all_weights returns a 4-component list:
        # ih weights, hh weights, ih bias, hh bias
        for layer in self.rnn.all_weights[0][0:2]:
            weights.append(np.array(layer.data))
        weights.append(getattr(self, 'fc1').weight.data.numpy())  # TODO MM update if multiple layers
        # weights.append([getattr(self, 'fc%d' % i).weight.data.numpy() for i in range(1, 3)])
        return weights


weights = [np.array([[0.14349649, 0.22849356, 0.0589799 ],
       [0.28085525, 0.33959095, 0.26693599],
       [0.29717143, 0.08791292, 0.13488955]]), np.array([[0.07945454, 0.08095541, 0.16352967],
       [0.06607006, 0.14902258, 0.03487133],
       [0.2830554 , 0.16441791, 0.05926117]]), np.array([[0.10590603, 0.08056413, 0.11820939],
       [0.08806567, 0.14830231, 0.02520716],
       [0.37728834, 0.16362323, 0.04283765]]), np.array([[0.0941876 , 0.0508151 , 0.11713735],
       [0.07832127, 0.09354034, 0.02497856],
       [0.33554163, 0.10320387, 0.04244916]]), np.array([[0.10338399, 0.09007417, 0.10102906],
       [0.08596848, 0.16580837, 0.0215436 ],
       [0.36830362, 0.18293782, 0.03661171]]), np.array([[0.09738886, 0.10750969, 0.14982824],
       [0.08098326, 0.19790366, 0.03194962],
       [0.34694606, 0.21834882, 0.05429594]]), np.array([[0.07963926, 0.04239418, 0.13476703],
       [0.06622366, 0.07803913, 0.02873794],
       [0.28371345, 0.08610125, 0.04883794]]), np.array([[0.09901042, 0.09007417, 0.09351238],
       [0.08233166, 0.16580837, 0.01994073],
       [0.35272284, 0.18293782, 0.03388776]]), np.array([[0.08120352, 0.04799111, 0.06678277],
       [0.06752442, 0.08834195, 0.01424087],
       [0.2892861 , 0.09746844, 0.02420127]]), np.array([0.7606715 , 0.24665178, 0.32597189])]
layer_sizes = [3,3,3,3,3,3,3,3,3,3]
plt.figure(figsize=(18,6))
ax = plt.gca()
plot_ann(layer_sizes, weights, ax=ax)

model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## TODO MM bug
# motion and color seem to be swapped sometimes
## e.g.:
# sample()
# For [-0.7030578, 1.1485194, 1.0], model thinks: 1, truth is: -1
# For [-1.4347728, 0.5917345, 1.0], model thinks: 1, truth is: -1
# For [-0.6270002, 0.7519773, 1.0], model thinks: 1, truth is: -1
# For [-0.4843572, 1.2000422, 1.0], model thinks: 1, truth is: -1

# TODO MM use cross entropy loss


def translate_to_cel(label):
    return nonzero(label == 1.0)[0]


def train_rnn():
    print('training model')
    time_start = time.perf_counter()
    num_data = 100
    mc_dataset = MotionColorDataset(num_data, 10)
    X, Y, _ = mc_dataset.get_xyz(num_data)
    X = np.array(X)
    Y = np.array(Y)  # TODO MM Y IS NOT TRUE LABEL! it's color label
    model.train()
    for epoch in range(num_epochs):
        hidden = model.init_hidden()
        output, hidden = model(torch.from_numpy(X), hidden)
        optimizer.zero_grad()
        loss = criterion(torch.squeeze(output), torch.from_numpy(Y))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{n_total_steps}], Loss: {loss.item() * 1000:.6f}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item() * 1000:.6f}')
            print(f'Time elapsed: {time.perf_counter() - time_start:0.2f} seconds')
    FILE = "model.pth"
    torch.save(model.state_dict(), FILE)
    model.eval()
    return model


# from model output, return -1, 0 or 1
def translate_output(x):
    classes = [-1, 0, 1]
    choice = []
    for i in x:
        prob = nn.functional.softmax(i[-1], dim=0).data
        choice.append(classes[torch.max(prob, dim=0)[1].item()])
    return choice


# dot format: [color, motion, context] where context=0 -> motion, context=1 -> color
def sample(model, debug=True):
    hidden = model.init_hidden(batch_size=1)
    dots, label = test_dataset[0]
    if debug:
        print('----')
    output, hidden = model(torch.from_numpy(np.array([dots])), hidden)
    translated_output = translate_output(output)
    if debug:
        print(f"For {dots}, model thinks: {translated_output}; true label: {label}")
    return output, label, (translated_output == label)


model = train_rnn()

# Test the model
# TODO MM fix this?
with torch.no_grad():
    n_correct = 0
    n_samples = 1000

    for i in range(n_samples):
        loss = 0.
        output, label, is_match = sample(model, debug=debug)
        loss += criterion(output.view(-1), torch.from_numpy(np.array(label)))
        if is_match:
            n_correct += 1
    acc = 100.0 * n_correct / n_samples
    avg_loss = loss / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc}%')
    print(f'Average loss on the {n_samples} test images: {avg_loss}')

# TODO MM
# X =
# Y =
# Z =
# params = [X, Y, Z]
# see datautils.py for how these are created
z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy = \
    analyze_info_flow_rnn(model, 'linear-svm')  # TODO MM corr for estimate via correlation
