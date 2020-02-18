import numpy as np

class RingBuffer(object):
    def __init__(self, length, shape):
        self.length = length
        
        if not isinstance(shape, list):
            shape = [shape]

        self.shape = shape
        self.val = np.zeros([length] + shape)
        self.curr_idx = 0
    
    def get(self, from_idx, to_idx=None):
        tmp_index = self.curr_idx
        self.curr_idx = 0
        result = self.get_relative(from_idx, to_idx)
        self.curr_idx = tmp_index
        return result

    '''
    gets the elements relative to the current index
    to-index is excluded, so for [0, 1, 2, 3] from 0 to 2 means [0,1]
    '''
    def get_relative(self, from_idx, to_idx=None):
        get_single = False

        if to_idx is None:
            get_single = True
            to_idx = from_idx+1

        if(to_idx < from_idx):
            return "to_idx < from_idx not allowed"
        if(to_idx == from_idx):
            return []

        from_idx += self.curr_idx
        to_idx += self.curr_idx

        from_idx %= self.length
        to_idx %= self.length

        if(from_idx > to_idx):
            result = np.concatenate([self.val[from_idx:], self.val[:to_idx]], axis=0)
        else:
            result = self.val[from_idx : to_idx]

        # If single time step should to be returned, remove time axis
        if get_single:
            result = result[-1,:]

        return result

    '''
    Maintains the current index
    '''
    def write(self, input, from_idx):

        size = input.shape[0]
        to_idx = from_idx + size

        from_idx %= self.length
        to_idx %= self.length

        # Just set the current index to from_idx and append
        tmp_curr_idx = self.curr_idx
        self.curr_idx = from_idx
        self.append(input)
        self.curr_idx = tmp_curr_idx


    def append_single(self, input):
        # input has shape (shape), so add the time axis
        new_input = input[np.newaxis, :]
        self.append(new_input)

    def append(self, input):
        input = np.asarray(input)  # Convert to numpy-array to use shape operations
        assert input[0].shape == self.val[0].shape, "Shape of Input to Ringbuffer does not fit."
        
        # input must be of shape (time_steps, dim) where time_steps is the number of to be appended elements
        size = input.shape[0]  # size = number of timesteps to add
        
        size_difference = (self.curr_idx + size) - self.length
        if(size_difference > 0):
            # split input
            split_point = self.length - self.curr_idx
            input_first = input[:split_point]
            input_second = input[split_point:]
        
            self.val[self.curr_idx :] = input_first
            self.val[0 : size_difference] = input_second

            print("##############################\nReached end of Ring buffer\n###########################")

        else:
            self.val[self.curr_idx : self.curr_idx + size] = input
            
        self.change_curr_idx(size)

    def change_curr_idx(self, num_time_steps):
        self.curr_idx = (self.curr_idx + num_time_steps) % self.length

    # def append(self, input, start_index):
    #     # just move the current index?
    #     self.curr_idx = self.length - (start_index - self.curr_idx)

# rb = RingBuffer(10, 1)
# rb.append(np.array([[0,1,2,3]]).transpose())
# print(rb.get(0,4))
# rb.append(np.array([[4,5]]).transpose())
# print(rb.get(0,6))
# rb.write(np.array([[2,4,6,8,10]]).transpose(), 1)
# print(rb.get(5,14))
# rb.append(np.array([[12,14,16,18,20]]).transpose())
# print(rb.get(0,9))




