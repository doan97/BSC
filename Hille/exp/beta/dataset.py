import numpy as np

import torch.utils.data as tud


class Dataset(tud.dataset.Dataset):
    def __init__(self):
        self.data = None
        self.current_series = None

    def reset(self):
        print('reset')
        self.data = []
        self.current_series = -1

    def start_new_series(self):
        self.data.append([])
        self.current_series += 1

    def transform_old_series(self):
        old_series = self.data[self.current_series]
        old_series = list(zip(*old_series))
        for idx, old_serie in enumerate(old_series):
            old_series[idx] = np.stack(list(old_serie))
        self.data[self.current_series] = old_series

    def append(self, transition):
        self.data[self.current_series].append(transition)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
