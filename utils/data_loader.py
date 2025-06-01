import numpy as np
from torch.utils.data import Dataset
import torch


class AccidentDataset(Dataset):
    def __init__(self, data_file, T_long, T_short, K, num_nodes, in_dim):
        """
        Args:
            data_file
            T_long
            T_short
        """

        self.data = np.fromfile(data_file, dtype=np.float32)
        self.data = self.data.reshape((-1, num_nodes, in_dim))
        self.T_long = T_long
        self.T_short = T_short
        self.K = K
        self.start_index = T_long
        self.end_index = self.data.shape[0] - K

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, index):
        t = index + self.start_index
        short_term_input = self.data[t - self.T_short : t]
        long_term_input = self.data[t - self.T_long : t]
        target = self.data[t + 1 : t + self.K + 1]
        short_term_input = torch.tensor(short_term_input, dtype=torch.float32)
        long_term_input = torch.tensor(long_term_input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return short_term_input, long_term_input, target
