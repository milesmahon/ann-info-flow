import time

import torch
import torch.nn as nn
import numpy as np
from numpy import nonzero

from analyze_info_flow import analyze_info_flow_rnn
from datasets.MotionColorDataset import MotionColorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 1
learning_rate = 0.0001
hidden_size = 4
num_layers = 1
batch_size = 10
input_size = 3  # (motion (float), color (float), context (bool/int))
output_size = 3  # (-1, 0, 1) one-hot encoded
debug = False

# Dataset params
train_dataset = MotionColorDataset(100, 10)
test_dataset = MotionColorDataset(100, 10)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
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
                                    # TODO MM include output (x2)? include input?
        return x2, hidden_i

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.hidden_size).to(device)

    def get_weights(self):
        # TODO MM verify this output makes sense
        weights = self.rnn.all_weights
        weights.append(getattr(self, 'fc1').weight.data.numpy())
        # weights.append([getattr(self, 'fc%d' % i).weight.data.numpy() for i in range(1, 3)])
        return weights


model = RNN(input_size, hidden_size, num_layers, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
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
    n_total_steps = 1000
    time_start = time.perf_counter()
    for epoch in range(num_epochs):
        for i, (dots, label) in enumerate(train_dataset):
            hidden = model.init_hidden()
            for dot in dots:
                output, hidden = model(torch.from_numpy(np.array([dot])), hidden)
            optimizer.zero_grad()
            # print(output.view(-1))
            # label = translate_to_cel(label)
            # print(torch.from_numpy(label))
            loss = criterion(output.view(-1), torch.from_numpy(label))
            loss.backward()
            optimizer.step()
            if i >= n_total_steps:
                break
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{n_total_steps}], Loss: {loss.item() * 1000:.6f}')
            print(f'Time elapsed: {time.perf_counter() - time_start:0.2f} seconds')
    FILE = "model.pth"
    torch.save(model.state_dict(), FILE)

    return model


# from model output, return -1, 0 or 1
def translate_output(x):
    classes = [-1, 0, 1]
    prob = nn.functional.softmax(x[-1], dim=0).data
    choice = classes[torch.max(prob, dim=0)[1].item()]
    return choice


# dot format: [color, motion, context] where context=0 -> motion, context=1 -> color
def sample(model, debug=True):
    hidden = model.init_hidden()
    dots, label = test_dataset[0]
    if debug:
        print('----')
    for dot in dots:
        output, hidden = model(torch.from_numpy(np.array([dot])), hidden)
        translated_output = translate_output(output)
        translated_label = translate_output(torch.from_numpy(label))
        if debug:
            print(f"For {dot}, model thinks: {translated_output}; true label: {translated_label}")
    return output, label, (translated_output == translated_label)


model = train_rnn()
n_samples = 1000

# Test the model
with torch.no_grad():
    n_correct = 0
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
analyze_info_flow_rnn(model, 'linear-svm')
