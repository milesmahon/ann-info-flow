from torch.utils.data import Dataset
import numpy as np


class MotionColorDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length

        # TODO just two random vars for now
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

    # returns seq_length samples from either the red or green distribution and the left/right distribution
    def __getitem__(self, index):
        coin_flip = np.random.binomial(1, 0.5)
        color_gen, color_label = (self.color_gen_g, -1) if coin_flip == 0 else (self.color_gen_r, 1)

        coin_flip = np.random.binomial(1, 0.5)
        motion_gen, motion_label = (self.motion_gen_right, -1) if coin_flip == 0 else (self.motion_gen_left, 1)
        out = []
        for i in range (self.seq_length):
            out.append([color_gen[i], motion_gen[i], np.float32(0)])

        # sequence, label
        # [seq_length x [3]], [2]
        # TODO in this config, model is learning both color and motion (both labels are provided)
        return out, [np.float32(color_label), np.float32(motion_label)]

    def __len__(self):
        return self.num_samples
