import torch
import numpy as np

class Manipulator:
    def __init__(self, velocity_noise=None, velocity_dropout=None):
        self.counter = 0
        self.noise = velocity_noise
        self.dropout = velocity_dropout

    def velocity_dropout(self, dropout_percent, inputs):
        self.counter += 1
        for idx, input in enumerate(inputs):
            dropout = True if np.random.random_sample() < dropout_percent else False
            data = input[0]
            if dropout:
                data[0] = 0
                data[1] = 0
                inputs[idx] = [data]
        return inputs

    def velocity_noise(self, noise_range, inputs):
        #noise range in percent
        for idx, input in enumerate(inputs):
            data = input[0]
            noise_value_x = noise_range * data[0]
            noise_value_y = noise_range * data[1]
            neg_x = 1 if np.random.random_sample() < 0.5 else -1
            neg_y = 1 if np.random.random_sample() < 0.5 else -1
            random_noise_x = neg_x * np.random.random_sample() * noise_value_x
            random_noise_y = neg_y * np.random.random_sample() * noise_value_y

            data[0] = data[0] + random_noise_x
            data[1] = data[1] + random_noise_y

            inputs[idx] = [data]
        self.counter += 1
        return inputs

    def manipulate(self, inputs):
        if self.noise is not None:
            inputs = self.velocity_noise(self.noise, inputs)

        if self.dropout is not None:
            inputs = self.velocity_dropout(self.dropout, inputs)

        return inputs




