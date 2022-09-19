from torch.utils.data import Dataset
import numpy as np


def one_hot(choice):
    return np.array([1, 0, 0], dtype=np.float32) if choice == -1 else np.array([0, 0, 1], dtype=np.float32)


class MotionColorDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length

        ### distribution variables
        ## color
        sigma_c = 0.5
        mu_red = -1
        mu_green = 1

        self.color_gen_r = np.float32(np.random.normal(mu_red, sigma_c, seq_length))
        self.color_gen_g = np.float32(np.random.normal(mu_green, sigma_c, seq_length))

        ## motion
        sigma_m = 0.5
        mu_right = 1
        mu_left = -1

        self.motion_gen_right = np.float32(np.random.normal(mu_right, sigma_m, seq_length))
        self.motion_gen_left = np.float32(np.random.normal(mu_left, sigma_m, seq_length))

    # returns seq_length samples from either the red or green distribution and the left or right distribution
    def __getitem__(self, index):
        # green or red (color)
        coin_flip = np.random.binomial(1, 0.5)
        color_gen, color_label = (self.color_gen_g, -1) if coin_flip == 0 else (self.color_gen_r, 1)

        # left or right (direction)
        coin_flip = np.random.binomial(1, 0.5)
        motion_gen, motion_label = (self.motion_gen_right, -1) if coin_flip == 0 else (self.motion_gen_left, 1)

        # color or motion (context)
        context = np.random.randint(0, 2)

        out = []
        for i in range(self.seq_length):
            out.append([color_gen[i], motion_gen[i], np.float32(context)])

        label = one_hot(color_label if context else motion_label)

        # sequence, label
        # [seq_length x [color, motion, context]], [one-hot encoding of the three classes (-1, 0, 1)]
        # color if context == 1 else motion
        return out, label

    def __len__(self):
        return self.num_samples
