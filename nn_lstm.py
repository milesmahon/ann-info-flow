import random
import re
import string
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import unidecode
import requests

# Download training data
url = "https://osf.io/f6z3p/download"
fname = "1522-0.txt"
r = requests.get(url, stream=True)
with open(fname, "wb") as f:
  f.write(r.content)
print(f"The file '{fname}' has been downloaded.")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(input, hidden)
        output = self.decoder(hidden.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)


# Turn string into list of longs
def char_tensor(str):
    tensor = torch.zeros(len(str)).long()
    for c in range(len(str)):
        tensor[c] = all_characters.index(str[c])
    return tensor


def evaluate(net, all_characters, prime_str, predict_len):
    hidden = net.init_hidden()
    predicted = prime_str

    # "Building up" the hidden state
    for p in range(len(prime_str) - 1):
        inp = char_tensor(prime_str[p])
        _, hidden = net(inp, hidden)

    # Tensorize of the last character
    inp = char_tensor(prime_str[-1])

    # For every index to predict
    for p in range(predict_len):
        # Pass the inputs to the model
        output, hidden = net(inp, hidden)

        # Pick the character with the highest probability
        top_i = torch.argmax(torch.softmax(output, dim=1))

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted


def grad_clipping(net, theta):
    """Clip the gradient."""
    params = [p for p in net.parameters() if p.requires_grad]

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    if norm > theta:
      for param in params:
        param.grad[:] *= theta / norm


# Single training step
def train_gen(inp, target, chunk_len):
    # Initialize hidden state, zero the gradients of decoder
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    # For each character in our chunk (except last), compute the hidden and ouput
    # Using each output, compute the loss with the corresponding target
    for c in range(chunk_len):
      output, hidden = decoder(inp[c], hidden)
      loss += criterion(output, target[c].unsqueeze(0))

    # Backpropagate, clip gradient and optimize
    loss.backward()
    grad_clipping(decoder, 1)
    decoder_optimizer.step()

    # Return average loss
    return loss.data.item() / chunk_len


n_epochs = 100000
hidden_size = 100
n_layers = 1
lr = 0.0005
print_every, plot_every = 500, 10


# Print a random chunk from the training data
def random_chunk(chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]


# Read the input training file - Julius Caesar
file = unidecode.unidecode(open(fname).read()).lower()
file = re.sub(r'[^a-z]+', ' ', file)
file_len = len(file)
print(f'\nLength of {fname} file is {file_len}')

chunk_len = 100
print(f"\nRandom chunk: {random_chunk(chunk_len)}")
all_characters = string.ascii_lowercase
all_characters += ' '
n_characters = len(all_characters)

# Create model, optimizer and loss function
decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

FILE = "model.pth"
decoder.load_state_dict(torch.load(FILE))
# model.eval()

all_losses = []
loss_avg = 0


# Get a random chunk from the training data,
# Convert its first n-1 chars into input char tensor
# Convert its last n-1 chars into target char tensor
def random_training_set(chunk_len):
    chunk = random_chunk(chunk_len)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target


sampleDecoder = RNN(27, 100, 27, 1)
text = evaluate(sampleDecoder, all_characters, 'hi', 10)
if text.startswith('hi') and len(text) == 12:
    print('Success!')
else:
    print('Need to change.')

# For every epoch
for epoch in range(1, n_epochs + 1):
    # Get a random (input, target) pair from training set and perform one training iteration
    loss = train_gen(*random_training_set(chunk_len), chunk_len)
    loss_avg += loss

    if epoch % print_every == 0:
      text = evaluate(decoder, all_characters, 'th', 50)
      print(f'Epoch {epoch} --------------------\n\t {text}')

    if epoch % plot_every == 0:
      all_losses.append(loss_avg / plot_every)
      loss_avg = 0

print('\n')
plt.figure()
plt.plot(all_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training loss for text generation')
plt.show()