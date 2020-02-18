import torch
import torch.nn as nn
import torch.optim
import torch.utils.data.dataloader as data_loader
import typing

import util


class Model:
    def __init__(self, model_config, seed=0):
        """
        TODO docstring

        Parameters
        -----------------------
        model_config: ModelConfig
            contains sizes for input, hidden, output layers and number of hidden layers
        """
        self.module = Module(model_config['input dimension'],
                             model_config['hidden dimension'],
                             model_config['number of layers'],
                             model_config['output dimension'])
        self.batch_size = model_config['batch size']
        self.initial_learning_rate = model_config['learning rate']
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=self.initial_learning_rate)
        self.loss = nn.MSELoss()
        self.loss_weightings = [model_config['position weight learning'],
                                model_config['acceleration weight learning'],
                                model_config['sensor weight learning']]

    def reset(self):
        """
        TODO docstring

        Returns
        -------

        """
        self.module = Module(self.module.input_dim,
                             self.module.hidden_dim,
                             self.module.num_layers,
                             self.module.output_dim)
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=self.initial_learning_rate)

    def get_hidden_state(self):
        """
        TODO docstring

        Returns
        -------

        """
        return self.module.get_hidden_state()

    def set_hidden_state(self, h_0, c_0):
        """
        TODO docstring

        Parameters
        ----------
        h_0
        c_0

        Returns
        -------

        """
        self.module.set_hidden_state(h_0, c_0)

    def learn(self, buffer, log_interval=10, single=False):
        """
        TODO docstring

        Parameters
        ----------
        buffer
        log_interval
        single

        Returns
        -------

        """
        if not buffer.is_full():
            return 0.0
        current_h0, current_c0 = self.module.get_hidden_state()
        current_h0 = current_h0.detach()
        current_c0 = current_c0.detach()
        new_data_loader = data_loader.DataLoader(dataset=buffer,
                                                 batch_size=self.batch_size,
                                                 shuffle=True)
        loss_sum = 0
        for idx, (h0, c0, input_sequence, target_sequence) in enumerate(new_data_loader):
            model_input = torch.cat(input_sequence, dim=2).float()

            model_target = [element.float() for element in target_sequence]

            split_sections = util.get_split_sections(target_sequence, 2)

            self.module.set_hidden_state(h0, c0)
            model_output = self.module(model_input)
            model_output = torch.split(model_output, split_sections, dim=2)

            loss = self.calculate_loss(
                model_output, model_target, )
            loss_sum += loss

            if (idx + 1) % log_interval == 0:
                print('batch {}/{} with loss {}'.format(idx + 1, len(new_data_loader), loss))
            if single:
                break
        print('' if single else loss_sum)
        self.module.set_hidden_state(current_h0, current_c0)
        return loss_sum

    def forward_batch(self, input_sequence):
        """
        TODO docstring

        Parameters
        ----------
        input_sequence

        Returns
        -------

        """
        return self.module(torch.cat(input_sequence, dim=2).float())

    def calculate_loss(self, output, target):
        """
        TODO docstring

        Parameters
        ----------
        output
        target

        Returns
        -------

        """
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for (w, o, t) in zip(self.loss_weightings, output, target):
            loss += w * self.loss(o, t)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        create pytorch module that later could encode the environment dynamics

        Parameters
        ----------
        input_dim : int
        hidden_dim : int
        num_layers : int
        output_dim : int
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.batch_size = 1

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        self.c_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        self.dense = nn.Linear(self.hidden_dim, self.output_dim)

    def get_hidden_state(self):
        """

        Returns
        -------
        h_0 : torch tensor
            of shape [batch_size, num_layers, hidden_size]
        c_0 : torch tensor
            of shape [batch_size, num_layers, hidden_size]

        """
        return self.h_0.permute(1, 0, 2), self.c_0.permute(1, 0, 2)

    def set_hidden_state(self, h_0, c_0):
        """

        Parameters
        ----------
        h_0 : torch tensor
            of shape [batch_size, num_layers, hidden_size]
        c_0 : torch tensor
            of shape [batch_size, num_layers, hidden_size]

        Returns
        -------

        """
        assert h_0.shape == c_0.shape
        self.batch_size = h_0.shape[0]
        self.h_0, self.c_0 = h_0.permute(1, 0, 2), c_0.permute(1, 0, 2)

    def get_input_shape(self):
        return self.input_dim

    def get_output_shape(self):
        return self.output_dim

    def forward(self, x):
        """
        compute forward pass through network

        Parameters
        ----------
        x : torch tensor
            of shape [batch_size, sequence_length, input_dimension]

        Returns
        -------
        x : torch tensor
            of shape [batch_size, sequence_length, output_dimension]

        """
        assert x.shape[0] == self.batch_size, "{}, {}".format(x, self.batch_size)
        x = x.permute(1, 0, 2)
        sequence_length = x.shape[0]
        batch_size = x.shape[1]
        x, (self.h_0, self.c_0) = self.lstm(x, (self.h_0, self.c_0))
        x = x.view(sequence_length * batch_size, self.hidden_dim)
        x = self.dense(x)
        x = x.view(sequence_length, batch_size, self.output_dim)
        x = x.permute(1, 0, 2)
        return x

    def __call__(self, *input_, **kwargs) -> typing.Any:
        # https://github.com/pytorch/pytorch/issues/24326
        return super().__call__(*input_, **kwargs)

