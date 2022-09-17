import torch
import torch.nn as nn
import numpy as np

# Device configuration
from datasets.MotionColorDataset import MotionColorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 1000
learning_rate = 0.0001
hidden_size = 128
num_layers = 1
batch_size = 100
input_size = 3  # (motion (float), color (float), context (bool/int))
output_size = 2  # (motion l/r, color r/g (bool/ints))

# Dataset params
train_dataset = MotionColorDataset(100, 10)
test_dataset = MotionColorDataset(100, 10)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    # if x is a batch, no need to pass hidden state beyond initial h0
    def forward(self, x, hidden):
        out, hidden_i = self.rnn(x, hidden)
        out = self.fc(out)
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
for epoch in range(num_epochs):
    for i, (dots, label) in enumerate(train_dataset):
        hidden = model.init_hidden()
        for dot in dots:
            output, hidden = model(torch.from_numpy(np.array([dot])), hidden)
        optimizer.zero_grad()
        loss = criterion(output.view(-1), torch.from_numpy(np.array(label)))
        # print(loss)
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item()*1000:.6f}')
        loss.backward()
        optimizer.step()
        if i >= n_total_steps:
            break

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
        if torch.equal(output.view(-1), torch.from_numpy(np.array(label))):
            n_correct += 1
            print(n_correct)
        loss += criterion(output.view(-1), torch.from_numpy(np.array(label)))
        if i == n_samples:
            break

    acc = 100.0 * n_correct / n_samples
    avg_loss = loss / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')
    print(f'Average loss on the {n_samples} test images: {avg_loss}')
