import numpy.random
from torch.utils.data import Dataset
import numpy as np
from numpy.random import default_rng


def one_hot(choice):
    return np.array([1, 0, 0], dtype=np.float32) if choice == -1 else np.array([0, 0, 1], dtype=np.float32)


class MotionColorDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length

        # distribution variables
        # color
        self.sigma_c = 0.5
        self.mu_red = -1
        self.mu_green = 1

        # motion
        self.sigma_m = 0.5
        self.mu_left = -1
        self.mu_right = 1

    # returns seq_length samples from either the red or green distribution and the left or right distribution
    def __getitem__(self, index):
        # green or red (color)
        rng = default_rng()
        coin_flip = rng.binomial(1, 0.5, 1)
        color_gen_r = np.float32(rng.normal(self.mu_red, self.sigma_c, self.seq_length))
        color_gen_g = np.float32(rng.normal(self.mu_green, self.sigma_c, self.seq_length))
        color_gen, color_label = (color_gen_r, -1) if coin_flip == 0 else (color_gen_g, 1)

        # left or right (direction)
        coin_flip = rng.binomial(1, 0.5, 1)
        motion_gen_l = np.float32(rng.normal(self.mu_left, self.sigma_m, self.seq_length))
        motion_gen_r = np.float32(rng.normal(self.mu_right, self.sigma_m, self.seq_length))
        motion_gen, motion_label = (motion_gen_l, -1) if coin_flip == 0 else (motion_gen_r, 1)

        # color or motion (context)
        context = np.float32(rng.binomial(1, 0.5, 1)[0])

        out = []
        for i in range(self.seq_length):
            out.append([color_gen[i], motion_gen[i], context])

        label = one_hot(motion_label if context else color_label)

        # sequence, label
        # [seq_length x [color, motion, context]], [one-hot encoding of the three classes (-1, 0, 1)]
        # motion if context == 1 else color
        return out, label

    def __len__(self):
        return self.num_samples
