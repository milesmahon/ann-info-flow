import time

import torch
import torch.nn as nn
import numpy as np

from datasets.MotionColorDataset import MotionColorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
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

## TODO
# why are labels not what I expect? (motion and color seem to be swapped)
## e.g.:
# sample()
# For [-0.7030578, 1.1485194, 1.0], model thinks: 1, truth is: -1
# For [-1.4347728, 0.5917345, 1.0], model thinks: 1, truth is: -1
# For [-0.6270002, 0.7519773, 1.0], model thinks: 1, truth is: -1
# For [-0.4843572, 1.2000422, 1.0], model thinks: 1, truth is: -1


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

# dot format: [color, motion, context] where context=0 -> motion, context=1 -> color
def sample(debug=True):
    hidden = model.init_hidden()
    dots, label = test_dataset[0]
    for dot in dots:
        output, hidden = model(torch.from_numpy(np.array([dot])), hidden)
        translated_output = translate_output(output)
        translated_label = translate_output(torch.from_numpy(label))
        if debug:
            print(f"For {dot}, model thinks: {translated_output}, truth is: {translated_label}")
    return output, label, (translated_output == translated_label)


n_samples = 1000

# Test the model
with torch.no_grad():
    n_correct = 0
    for i in range(n_samples):
        loss = 0.
        output, label, is_match = sample(debug=False)
        loss += criterion(output.view(-1), torch.from_numpy(np.array(label)))
        if is_match:
            n_correct += 1
    acc = 100.0 * n_correct / n_samples
    avg_loss = loss / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')
    print(f'Average loss on the {n_samples} test images: {avg_loss}')
