import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
import numpy as np

# Device configuration
from datasets.ShakespeareDataset import ShakespeareDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset hyperparams
#
train_dataset = ShakespeareDataset()
test_dataset = ShakespeareDataset()

output_size = train_dataset.output_size
batch_size = 100
input_size = train_dataset.x.size()[2]
sequence_length = train_dataset.x.size()[1]

# input size: size of a single character when one-hot encoded (62)
# sequence_length: number of characters in one sentence (83)


# output_size = 10
# batch_size = 100
# input_size = 28
# sequence_length = 28

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='./data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='./data',
#                                           train=False,
#                                           transform=transforms.ToTensor())

# Hyper-parameters
num_epochs = 1000
learning_rate = 0.001
hidden_size = 128
num_layers = 2

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# If manually doing step-wise input to the RNN, (e.g., character-by-character),
#   passing hidden state at each step is required
#   if passing an entire sequence at once, the RNN module will take care of it.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, hidden = self.lstm(x, (h0, c0))
        # TODO pass hidden state or output to fc?
        out = self.fc(out)
        # out: (batch_size, seq_length, output_size)
        return out


model = RNN(input_size, hidden_size, num_layers, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images = images.reshape(-1, sequence_length, input_size).to(device)  # MNIST
        images = images.to(device)  # Shakespeare
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use 100 for MNIST; Shakespeare has 3299 sentences = 33 batches
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

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

def predict(char):
    char = train_dataset.one_hot_encode(char)
    out = model(torch.from_numpy(np.array([[char]])))
    prob = nn.functional.softmax(out[-1][-1], dim=0).data
    # print(prob)
    char_ind = torch.max(prob, dim=0)[1].item()
    # print(char_ind)
    return train_dataset.int_to_char[char_ind]


prompt = "hey"
resp_length = 25
while (prompt != "end"):
    prompt = input("Say:")
    sentence = ''
    predicted_char = ''
    for char in prompt:
        predicted_char = predict(char)
    for i in range(resp_length):
        sentence += predicted_char
        predicted_char = predict(predicted_char)
    print(sentence)


