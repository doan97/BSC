# with minor changes taken from
# https://github.com/eric-wieser/numpy_ringbuffer/blob/master/numpy_ringbuffer/__init__.py

"""
Ring (aka circular) buffers backed by a numpy array, supporting:
    - Operations from collections.deque
        b.append(val)
        b.appendleft(val)
        b.extend(val)
        b.extendleft(val)
        b.pop(val)
        b.popleft(val)
    - The collections.Sequence protocol (unoptimized)
    - C-side unwrapping into an array with np.array(b)
    - Arbitrary element dtypes, including extra dimensions like RingBuffer(N, data_type=(int, 3))
"""


import numpy as np
import torch.utils.data as data


class Memory(data.Dataset):
    def __init__(self, capacity, data_type=float, allow_overwrite=True):
        """
        Create a new ring buffer with the given capacity and element type
        Parameters
        ----------
        capacity: int
            The maximum capacity of the ring buffer
        data_type: data-type, optional
            Desired type of buffer elements. Use a type like (float, 2) to
            produce a buffer with shape (N, 2)
        allow_overwrite: bool
            If false, throw an IndexError when trying to append to an already
            full buffer
        """
        super(Memory, self).__init__()
        self._arr = np.empty(capacity, data_type)
        self._left_index = 0
        self._right_index = 0
        self._capacity = capacity
        self._data_type = data_type
        self._allow_overwrite = allow_overwrite

    def clear(self):
        """loose every entry of memory instead of creating new object"""
        self._arr = np.empty(self._capacity, self._data_type)
        self._left_index = 0
        self._right_index = 0

    @property
    def is_full(self):
        """True if there is no more space in the buffer """
        return len(self) == self._capacity

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def shape(self):
        return (len(self),) + self._arr.shape[1:]

    @property
    def maxlen(self):
        return self._capacity

    def _unwrap(self):
        """Copy the data from this buffer into unwrapped form """
        return np.concatenate((
            self._arr[self._left_index:min(self._right_index, self._capacity)],
            self._arr[:max(self._right_index - self._capacity, 0)]
        ))

    def __array__(self):
        return self._unwrap()

    def _fix_indices(self):
        """
        Enforce our invariant that 0 <= self._left_index < self._capacity
        """
        if self._left_index >= self._capacity:
            self._left_index -= self._capacity
            self._right_index -= self._capacity
        elif self._left_index < 0:
            self._left_index += self._capacity
            self._right_index += self._capacity

    def append(self, value):
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError('append to a full RingBuffer with overwrite disabled')
            elif not len(self):
                return
            else:
                self._left_index += 1

        self._arr[self._right_index % self._capacity] = value
        self._right_index += 1
        self._fix_indices()

    def appendleft(self, value):
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError('append to a full RingBuffer with overwrite disabled')
            elif not len(self):
                return
            else:
                self._right_index -= 1

        self._left_index -= 1
        self._fix_indices()
        self._arr[self._left_index] = value

    def pop(self):
        if len(self) == 0:
            raise IndexError("pop from an empty RingBuffer")
        self._right_index -= 1
        self._fix_indices()
        res = self._arr[self._right_index % self._capacity]
        return res

    def popleft(self):
        if len(self) == 0:
            raise IndexError("pop from an empty RingBuffer")
        res = self._arr[self._left_index]
        self._left_index += 1
        self._fix_indices()
        return res

    def extend(self, values):
        lv = len(values)
        if len(self) + lv > self._capacity:
            if not self._allow_overwrite:
                raise IndexError('extend a RingBuffer such that it would overflow, with overwrite disabled')
            elif not len(self):
                return
        if lv >= self._capacity:
            # wipe the entire array! - this may not be thread safe
            self._arr[...] = values[-self._capacity:]
            self._right_index = self._capacity
            self._left_index = 0
            return

        ri = self._right_index % self._capacity
        sl1 = np.s_[ri:min(ri + lv, self._capacity)]
        sl2 = np.s_[:max(ri + lv - self._capacity, 0)]
        self._arr[sl1] = values[:sl1.stop - sl1.start]
        self._arr[sl2] = values[sl1.stop - sl1.start:]
        self._right_index += lv

        self._left_index = max(self._left_index, self._right_index - self._capacity)
        self._fix_indices()

    def extendleft(self, values):
        lv = len(values)
        if len(self) + lv > self._capacity:
            if not self._allow_overwrite:
                raise IndexError('extend a RingBuffer such that it would overflow, with overwrite disabled')
            elif not len(self):
                return
        if lv >= self._capacity:
            # wipe the entire array! - this may not be thread safe
            self._arr[...] = values[:self._capacity]
            self._right_index = self._capacity
            self._left_index = 0
            return

        self._left_index -= lv
        self._fix_indices()
        li = self._left_index
        sl1 = np.s_[li:min(li + lv, self._capacity)]
        sl2 = np.s_[:max(li + lv - self._capacity, 0)]
        self._arr[sl1] = values[:sl1.stop - sl1.start]
        self._arr[sl2] = values[sl1.stop - sl1.start:]

        self._right_index = min(self._right_index, self._left_index + self._capacity)

    def __len__(self):
        return self._right_index - self._left_index

    def __getitem__(self, item):
        # handle simple (b[1]) and basic (b[np.array([1, 2, 3])]) fancy indexing specially
        if not isinstance(item, tuple):
            item_arr = np.asarray(item)
            if issubclass(item_arr.dtype.type, np.integer):
                item_arr = (item_arr + self._left_index) % self._capacity
                return self._arr[item_arr]

        # for everything else, get it right at the expense of efficiency
        return self._unwrap()[item]

    def __iter__(self):
        # alarmingly, this is comparable in speed to using itertools.chain
        return iter(self._unwrap())

    # Everything else
    def __repr__(self):
        return '<RingBuffer of {!r}>'.format(np.asarray(self))