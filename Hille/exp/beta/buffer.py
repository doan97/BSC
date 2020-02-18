import collections

import numpy as np
import torch.utils.data.dataset as tud

import util


class RingBuffer(tud.Dataset):
    def __init__(self, max_length, sequence_length, shapes, keys, sample_true_trajectories=False):
        assert max_length >= sequence_length
        self.deque = collections.deque(maxlen=max_length)
        self.sequence_length = sequence_length
        self.shapes = shapes
        self.keys = keys
        self.sample_true_trajectories = sample_true_trajectories
        self.sample_indices = collections.deque()
        self.seen_items = 0
        self.index_of_last_added_item = None
        super().__init__()

    def is_full(self):
        return len(self.deque) == self.deque.maxlen

    def append(self, item):
        assert set(item.keys()) == set(self.keys)
        # TODO shape check
        # assert stuff
        self.deque.append(item)
        self.index_of_last_added_item = self.seen_items % len(self.deque)
        self.seen_items += 1

    def extend(self, items):
        for item in items:
            self.append(item)

    def get_last_item(self):
        return self.deque[-1]

    def slice_deque(self, start, end):
        deque_slice = [np.empty((end-start, *shape)) for shape in self.shapes]
        for i in range(start, end):
            idx = i % len(self.deque)
            item = self.deque[idx]
            for j, key in enumerate(self.keys):
                deque_slice[j][i-start] = np.array(item[key])
        return deque_slice

    def get_sequences(self, deque_slice, target_too=False):
        input_sequence = []
        target_sequence = []
        for j, key in enumerate(self.keys):
            if key == 'position':
                pos_delta = util.get_differences(deque_slice[j])
                input_sequence.append(pos_delta[:-1])
                if target_too:
                    target_sequence.append(pos_delta[1:])
            elif key == 'velocity':
                vel_delta = util.get_differences(deque_slice[j])
                input_sequence.append(vel_delta[:-1])
                if target_too:
                    target_sequence.append(vel_delta[1:])
            elif key == 'sensor readings':
                input_sequence.append(deque_slice[j][1:-1])
                if target_too:
                    target_sequence.append(deque_slice[j][2:])
            elif key == 'motor commands':
                input_sequence.append(deque_slice[j][1:-1])
        if target_too:
            return input_sequence, target_sequence
        else:
            return input_sequence

    def __getitem__(self, idx):
        deque_slice = self.slice_deque(idx-1, idx + self.sequence_length + 1)
        input_sequence, target_sequence = self.get_sequences(deque_slice, True)
        return \
            self.deque[idx - 1]['hidden state h0'], \
            self.deque[idx - 1]['hidden state c0'], \
            input_sequence,\
            target_sequence

    def __len__(self):
        return len(self.deque)
