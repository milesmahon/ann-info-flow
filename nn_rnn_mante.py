import torch
import torch.nn as nn
import numpy as np

# Device configuration
from datasets.MotionColorDataset import MotionColorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
learning_rate = 0.001
hidden_size = 128
num_layers = 1
batch_size = 100
input_size = 1  # TODO for single random var in/out  use 1/1
output_size = 1

# Dataset params
train_dataset = MotionColorDataset(100, 10)
test_dataset = MotionColorDataset(100, 10)

# Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    # if x is a batch, no need to pass hidden state beyond initial h0
    def forward(self, x, hidden):
        # Set initial hidden states (and cell states for LSTM)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        # out = self.softmax(out) # TODO this?
        # out: (batch_size, seq_length, output_size)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.hidden_size).to(device)


# input_size = 3  # (motion (float), color (float), context (bool/int))
# output_size = 2  # (motion l/r, color r/g (bool/ints))

model = RNN(input_size, hidden_size, num_layers, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('start')
# Train the model
n_total_steps = 1000
for epoch in range(num_epochs):
    for i, (dots, label) in enumerate(train_dataset):
        optimizer.zero_grad()
        hidden = model.init_hidden()
        for dot in dots:
            output, hidden = model(torch.from_numpy(np.array([[dot]])), hidden)
            print(output, label)
        loss = criterion(output.view(-1), torch.from_numpy(np.array(np.float32(label))))
        # print(loss)
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()
        if i >= n_total_steps:
            break

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# MNIST
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         # images = images.reshape(-1, sequence_length, input_size).to(device)  # needed for MNIST
#         images = images.to(device)  # Shakespeare
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value, index)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         print(predicted.size())
#         print(labels.size())
#         n_correct += (predicted == labels).sum().item()
#
#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')

FILE = "model.pth"
torch.save(model.state_dict(), FILE)
