from torch.utils.data import Dataset
import torch
import numpy as np

# Designed for many-to-many predictions, wherein a single sequence (of length e.g. 83), yields an equivalently-sized
# output.
class ShakespeareDataset(Dataset):
    def __init__(self):
        with open('shakespeare.txt') as f:
            training_data = f.readlines()
            self.x, self.y, self.n_samples, self.output_size, self.int_to_char, self.char_to_int = self.process_input(training_data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def process_input(self, training_data):
        # unique characters in training data
        chars = set(''.join(training_data))

        # pad shorter strings with whitespace for consistent sequence length
        max_len = len(max(training_data, key=len))
        for i in range(len(training_data)):
            while len(training_data[i]) < max_len:
                training_data[i] += ' '

        # characters to integer values and vice-versa
        int_to_char = dict(enumerate(chars))
        char_to_int = {char: index for index, char in int_to_char.items()}

        # convert training data to input and target sequences
        # e.g.,
        # input: It is the east, and Juliet is the su
        # target: t is the east, and Juliet is the sun
        input_sequences = [training_data[i][:-1] for i in range(len(training_data))]
        target_sequences = [training_data[i][1:] for i in range(len(training_data))]

        # one-hot encode sequences
        dict_size = len(chars)
        for i in range(len(training_data)):
            input_sequences[i] = [self.one_hot_encode(char, char_to_int) for char in input_sequences[i]]
            target_sequences[i] = [self.one_hot_encode(char, char_to_int) for char in target_sequences[i]]

        x = torch.Tensor(np.array(input_sequences))
        y = torch.Tensor(np.array(target_sequences))
        n_samples = len(training_data)

        return x, y, n_samples, len(chars), int_to_char, char_to_int

    def one_hot_encode(self, char, char_to_int=None):
        if char_to_int is None:
            char_to_int = self.char_to_int
        features = np.zeros(len(char_to_int), dtype=np.float32)
        features[char_to_int[char]] = 1
        return features

    def one_hot_decode(self, array):
        int_char = array.index(1)
        return self.int_to_char[int_char]