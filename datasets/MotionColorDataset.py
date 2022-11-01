from torch.utils.data import Dataset
from numpy.random import default_rng
import numpy as np
import math
from scipy.stats import norm
import random

def one_hot(choice):
    # return np.array([1, 0, 0], dtype=np.float32) if choice == -1 else np.array([0, 0, 1], dtype=np.float32)
    return [1, 0] if choice == -1 else [0, 1]


def des_acc_to_sigma(seq_length, des_acc):
    return math.sqrt(seq_length) / norm.ppf(des_acc)


class MotionColorDataset(Dataset):
    def __init__(self, num_samples, seq_length, desired_acc=0.95):
        self.num_samples = num_samples
        self.seq_length = seq_length

        # desired ideal average accuracy of the network determines stddev of distributions.
        # assuming mean of +-1, this is based on the probability of a sample being negative when the mean is positive,
        # and vice versa.
        self.sigma_m = self.sigma_c = des_acc_to_sigma(seq_length, desired_acc)

        # distribution variables
        # color
        self.mu_red = -1
        self.mu_green = 1

        # motion
        self.mu_left = -1
        self.mu_right = 1

    def gen_color(self, sigma=None):
        rng = default_rng()
        if sigma is not None:
            self.sigma_c = sigma
        coin_flip = rng.binomial(1, 0.5, 1)
        color_gen_r = rng.normal(self.mu_red, self.sigma_c, self.seq_length)
        color_gen_g = rng.normal(self.mu_green, self.sigma_c, self.seq_length)
        return (color_gen_r, 0) if coin_flip == 0 else (color_gen_g, 1)

    def gen_motion(self, sigma=None):
        rng = default_rng()
        if sigma is not None:
            self.sigma_m = sigma
        coin_flip = rng.binomial(1, 0.5, 1)
        motion_gen_l = rng.normal(self.mu_left, self.sigma_m, self.seq_length)
        motion_gen_r = rng.normal(self.mu_right, self.sigma_m, self.seq_length)
        return (motion_gen_l, 0) if coin_flip == 0 else (motion_gen_r, 1)

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
        label = motion_label if context else color_label

        # sequence, label
        # [seq_length x [color, motion, context]], [one-hot encoding of the three classes (-1, 0, 1)]
        # motion if context == 1 else color
        return out, label

    def __len__(self):
        return self.num_samples

    # X is same format as in __getitem__() (batch_size (100-1000) x seq length (10) x input size (3))
    # Y is color label
    # Z is motion label
    # context = 0 or 1, to set all context to color or motion (respectively). default none
    # context_time = pro, retro, or none/always for when context should be supplied. default always
    # vary_acc = T/F. vary accuracy between trials. default false
    def get_xyz(self, num_samples, context=None, context_time="always", vary_acc=False):
        X = []  # input (size 3)
        Y = []  # color label (size 1)
        Z = []  # motion label
        U = []  # true label/correct output
        C = []  # context label

        coherence = [0.7, 0.85, 0.99]  # if vary_acc
        if context is not None:
            context_values = [context] * num_samples
        else:
            context_values = np.random.randint(0, 2, size=num_samples)
        for i in range(num_samples):
            if vary_acc:  # vary desired accuracy between trials
                color, color_label = self.gen_color(sigma=des_acc_to_sigma(self.seq_length, random.choice(coherence)))
                motion, motion_label = self.gen_motion(sigma=des_acc_to_sigma(self.seq_length, random.choice(coherence)))
            else:
                color, color_label = self.gen_color()
                motion, motion_label = self.gen_motion()
            context = context_values[i]
            if context_time == "pro":
                dots = [[m, c, -1] for m, c in zip(color, motion)]  # -1 corresponds to no context
                dots[0][2] = context  # prospective context, only on first dot
                X.append(dots)
            elif context_time == "retro":
                dots = [[m, c, -1] for m, c in zip(color, motion)]
                dots[-1][2] = context  # retrospective context, only on last dot
                X.append(dots)
            else:
                X.append([[m, c, context] for m, c in zip(color, motion)])  # always
            Y.append(color_label)
            Z.append(motion_label)
            U.append(motion_label if context else color_label)
            C.append(context)
        return X, Y, Z, U, C
