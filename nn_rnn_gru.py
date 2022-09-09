import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms

# Device configuration
from datasets.ShakespeareDataset import ShakespeareDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset hyperparams # TODO fix these for shakespeare set
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
num_epochs = 2
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


# Fully connected neural network with one hidden layer
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

        # Forward propagate RNN
        # out, _ = self.rnn(x, h0)
        # or:
        out, _ = self.lstm(x, (h0, c0))

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

        # use 100 for MNIST, Shakespeare has 3299 sentences = 33 batches
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

# FILE = "model.pth"
# torch.save(model, FILE)


# This function takes in the model and character as arguments and returns the next character prediction and hidden state
# def predict(model, character):
#     # One-hot encoding our input to fit into the model
#     character = np.array([[char_decode[c] for c in character]])
#     character = one_hot_encode(character, dict_size, character.shape[1], 1)
#     character = torch.from_numpy(character)
#     character.to(device)
#
#     out, hidden = model(character)
#
#     prob = nn.functional.softmax(out[-1], dim=0).data
#     # Taking the class with the highest probability score from the output
#     char_ind = torch.max(prob, dim=0)[1].item()
#
#     return char_encode[char_ind], hidden
#
#
# # This function takes the desired output length and input characters as arguments, returning the produced sentence
# def sample(model, out_len, start='hey'):
#     model.eval() # eval mode
#     start = start.lower()
#     # First off, run through the starting characters
#     chars = [ch for ch in start]
#     size = out_len - len(chars)
#     # Now pass in the previous characters and get a new one
#     for ii in range(size):
#         char, h = predict(model, chars)
#         chars.append(char)
#
#     return ''.join(chars)

prompt = "hey"
while (prompt != "end"):
    prompt = input("Say:")
    sentence = []
    for char in prompt:
        sentence += train_dataset.one_hot_encode(char)
    resp = model([sentence])
    print(resp)

    prob = nn.functional.softmax(resp[-1], dim=0).data
    # train_dataset.int_to_char(torch.max(prob, dim=0)[1].item())
    # print(''.join(out))


