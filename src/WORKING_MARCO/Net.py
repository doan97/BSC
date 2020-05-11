import typing


import torch
import torch.nn as nn


class Net(nn.Module):
    # PyTorch network class
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size=1, num_layers=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.dense = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self):
        # i: input, o: output, h: hidden outs, c:
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),  # h (output to next cell)
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))  # c (inner Cell Value)

    def forward(self, input_, hidden_=None):
        # Forward pass through LSTM layer
        # shape of all_h: [input_size, batch_size, hidden_dim]
        # shape of last_output: (h, c), where h and c both
        # have shape (num_layers, batch_size, hidden_dim).
        if hidden_ is None:
            all_h, last_output = self.lstm(input_)
        else:
            all_h, last_output = self.lstm(input_, hidden_)

        all_predictions = self.dense(all_h[:, -1, :])

        return all_predictions, last_output

    def __call__(self, *input_, **kwargs) -> typing.Any:
        # https://github.com/pytorch/pytorch/issues/24326
        return super().__call__(*input_, **kwargs)
