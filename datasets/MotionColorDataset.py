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
        # red
        mu_red = -1
        # green
        mu_green = 1

        self.color_gen_r = np.float32(np.random.normal(mu_red, sigma_c, seq_length))
        self.color_gen_g = np.float32(np.random.normal(mu_green, sigma_c, seq_length))

    # returns seq_length samples from either the red or green distribution
    def __getitem__(self, index):
        coin_flip = np.random.binomial(1, 0.5)
        if coin_flip == 0:
            return self.color_gen_g, -1
        else:
            return self.color_gen_r, 1

    def __len__(self):
        return self.num_samples
