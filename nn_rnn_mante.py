import time

import torch
import torch.nn as nn
import numpy as np

# Device configuration
from torch.utils.data import DataLoader

from datasets.MotionColorDataset import MotionColorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 100
learning_rate = 0.0001
hidden_size = 128
num_layers = 1
batch_size = 100
input_size = 3  # (motion (float), color (float), context (bool/int))
output_size = 3  # (-1, 0, 1)

# Dataset params
train_dataset = MotionColorDataset(100, 10)
test_dataset = MotionColorDataset(100, 10)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    # if x is a batch, no need to pass hidden state beyond initial h0
    def forward(self, x, hidden):
        out, hidden_i = self.rnn(x, hidden)
        out = self.fc1(out)
        out = self.fc2(out)
        return out, hidden_i

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.hidden_size).to(device)


model = RNN(input_size, hidden_size, num_layers, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
print('training model')
n_total_steps = 1000
time_start = time.perf_counter()
for epoch in range(num_epochs):
    for i, (dots, label) in enumerate(train_dataset):
        hidden = model.init_hidden()
        for dot in dots:
            output, hidden = model(torch.from_numpy(np.array([dot])), hidden)
        optimizer.zero_grad()
        loss = criterion(output.view(-1), torch.from_numpy(label))
        loss.backward()
        optimizer.step()
        if i >= n_total_steps:
            break
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{n_total_steps}], Loss: {loss.item() * 1000:.6f}')
        print(f'Time elapsed: {time.perf_counter() - time_start:0.2f} seconds')


# from model output, return -1, 0 or 1
def translate_output(x):
    classes = [-1, 0, 1]
    prob = nn.functional.softmax(x[-1], dim=0).data
    choice = classes[torch.max(prob, dim=0)[1].item()]
    return choice


FILE = "model.pth"
torch.save(model.state_dict(), FILE)

# Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 1000
    loss = 0.
    for i, (dots, label) in enumerate(test_dataset):
        hidden = model.init_hidden()
        for dot in dots:
            output, hidden = model(torch.from_numpy(np.array([dot])), hidden)
        if translate_output(output) == translate_output(torch.from_numpy(label)):
            n_correct += 1
        loss += criterion(output.view(-1), torch.from_numpy(np.array(label)))
        if i == n_samples:
            break

    acc = 100.0 * n_correct / n_samples
    avg_loss = loss / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')
    print(f'Average loss on the {n_samples} test images: {avg_loss}')
