#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dimensions, num_layers):
        super(RNN, self).__init__()
        self.hidden_dimensions = hidden_dimensions
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_dimensions, num_layers, batch_first=True)
        # Fully connected layer for converting RNN output to desired output size
        self.fc = nn.Linear(hidden_dimensions, output_size)

    def init_hidden_layer(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dimensions)
        # print(hidden)
        return hidden

    def forward(self, x):
        # TODO why not pass in hidden? where is hidden state stored/why isn't it?
        batch_size = x.size(0)
        hidden = self.init_hidden_layer(batch_size)
        output, hidden = self.rnn(x, hidden)

        # reshape output to fit into fc layer
        output = output.contiguous().view(-1, self.hidden_dimensions)
        output = self.fc(output)

        return output, hidden

# training_data = ['test', 'hello', 'how are you', 'thanks for having me', 'good', 'great']

with open('shakespeare.txt') as f:
    training_data = f.readlines()

# unique characters in training data
chars = set(''.join(training_data))

# characters to integer values and vice-versa
char_encode = dict(enumerate(chars))
char_decode = {char: index for index, char in char_encode.items()}

# prepare to batch training data by padding shorter strings with whitespace
max_len = len(max(training_data, key=len))
for i in range(len(training_data)):
    while len(training_data[i]) < max_len:
        training_data[i] += ' '

# convert training data to input and target sequences
input_sequences = [training_data[i][:-1] for i in range(len(training_data))]
target_sequences = [training_data[i][1:] for i in range(len(training_data))]

# one-hot encode sequences
for i in range(len(training_data)):
    input_sequences[i] = [char_decode[char] for char in input_sequences[i]]
    target_sequences[i] = [char_decode[char] for char in target_sequences[i]]

dict_size = len(chars)
sequence_length = max_len - 1
batch_size = len(training_data)

def one_hot_encode(sequence, dict_size, sequence_length, batch_size):
    # (batch-size length) list of matrices (of dimension sequence_length x dict_size), where dict_size is the length of the one-hot array
    features = np.zeros((batch_size, sequence_length, dict_size), dtype=np.float32)

    # for each phrase
    for i in range(batch_size):
        # for each character in phrase
        for j in range(sequence_length):
            # create one-hot encoded array
            features[i, j, sequence[i][j]] = 1
    return features

input_sequences = one_hot_encode(input_sequences, dict_size, sequence_length, batch_size)

input_sequences = torch.from_numpy(input_sequences)
target_sequences = torch.Tensor(target_sequences)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = RNN(input_size=dict_size, output_size=dict_size, hidden_dimensions=128, num_layers=3)
model.to(device)

n_epochs = 1000
learning_rate = 0.01

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char_decode[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character.to(device)

    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return char_encode[char_ind], hidden


# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)


prompt="what"
num_chars=25

for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()
    input_sequences.to(device)
    output, hidden = model(input_sequences) # TODO does this call forward?
    # print(target_sequences.view(-1))
    loss = loss_function(output, target_sequences.view(-1).long()) # TODO .view again
    loss.backward()
    optimizer.step()

    if epoch%10 == 0:
        print(epoch)
        print(loss.item())

    if epoch % 100 == 0:
        print(sample(model, num_chars, prompt))

while (prompt != "end"):
    prompt = input("Say:")
    print(sample(model, num_chars, prompt))

# sample(model, 15, 'good')
