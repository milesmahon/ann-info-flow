from torch.utils.data import Dataset
from numpy.random import default_rng
import numpy as np


def one_hot(choice):
    # return np.array([1, 0, 0], dtype=np.float32) if choice == -1 else np.array([0, 0, 1], dtype=np.float32)
    return [1, 0] if choice == -1 else [0, 1]


class MotionColorDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length

        # distribution variables
        # color
        self.sigma_c = 1.75
        self.mu_red = -1
        self.mu_green = 1

        # motion
        self.sigma_m = 1.75
        self.mu_left = -1
        self.mu_right = 1

    def gen_color(self):
        rng = default_rng()
        coin_flip = rng.binomial(1, 0.5, 1)
        color_gen_r = rng.normal(self.mu_red, self.sigma_c, self.seq_length)
        color_gen_g = rng.normal(self.mu_green, self.sigma_c, self.seq_length)
        return (color_gen_r, 0) if coin_flip == 0 else (color_gen_g, 1)

    def gen_motion(self):
        rng = default_rng()
        coin_flip = rng.binomial(1, 0.5, 1)
        motion_gen_l = rng.normal(self.mu_left, self.sigma_m, self.seq_length)
        motion_gen_r = rng.normal(self.mu_right, self.sigma_m, self.seq_length)
        return (motion_gen_l, 0) if coin_flip == 0 else (motion_gen_r, 1)  # TODO using 0 for left, 1 for right to fit with mutual info code

    # returns seq_length samples from either the red or green distribution and the left or right distribution
    def __getitem__(self, index):
        rng = default_rng()
        # green or red (color)
        color_gen, color_label = self.gen_color()

        # left or right (direction)
        motion_gen, motion_label = self.gen_motion()

        # color or motion (context)
        context = rng.binomial(1, 0.5, 1)[0]

        out = []
        for i in range(self.seq_length):
            out.append([color_gen[i], motion_gen[i], context])

        # color = 0, motion = 1
        label = motion_label if context else color_label  # TODO not one-hot encoding this anymore, for MI code

        # sequence, label
        # [seq_length x [color, motion, context]], [one-hot encoding of the three classes (-1, 0, 1)]
        # motion if context == 1 else color
        return out, label

    def __len__(self):
        return self.num_samples

    # For testing information flow, context always indicates color (0.0 as context value). This is to isolate
    #   information flow analysis to color vs motion rather than bias vs accuracy.
    # X is same format as in __getitem__()
    # Y is one-hot encoded color label
    # Z is motion label
    def get_xyz(self, num_samples, context=None):
        X = []
        Y = []
        Z = []
        U = []
        if context is not None:
            context_values = [context] * num_samples
        else:
            context_values = np.random.randint(0, 2, size=num_samples)
        for i in range(num_samples):
            color, color_label = self.gen_color()
            motion, motion_label = self.gen_motion()
            context = context_values[i]
            X.append([[m, c, context] for m, c in zip(color, motion)])
            Y.append(color_label)  # TODO not one-hot encoding this anymore, for MI code
            Z.append(motion_label)
            U.append(motion_label if context else color_label)
        return X, Y, Z, U
