import numpy as np
import torch


class Inference:
    def __init__(self, inference_config, model, motor_commands_dimensions, seed=0):
        """
        TODO docstring

        Parameters
        ----------
        inference_config
        model
        motor_commands_dimensions
        seed
        """
        self.prediction_horizon = inference_config['prediction horizon']
        self.inference_iterations = inference_config['inference iterations']
        self.step_size = inference_config['step size']

        self.position_loss_weight = inference_config['position loss weight']
        self.sensor_readings_loss_weight = inference_config['sensor reading loss weight']

        self.motor_commands_dimensions = motor_commands_dimensions

        self.model = model
        self.loss = torch.nn.MSELoss()

        self.input_shape = None
        self.output_shape = None
        self.mask = None

        self.start_position = None
        self.target_position = None

        self.position_delta = None
        self.velocity_delta = None
        self.sensor_readings = None

        self.inputs = None
        self.outputs = None

        self.h_0 = None
        self.c_0 = None
        self.predictions = None
        self.action = None

    def reset(self):
        """
        TODO docstring

        Returns
        -------

        """
        self.action = None

    def set_next_motor_command(self, motor_command):
        """
        TODO docstring

        Parameters
        ----------
        motor_command : (motor_commands_dimensions,) array

        Returns
        -------

        """
        self.inputs[0, 0, 0, -4:] = torch.tensor(motor_command)

    def get_next_motor_command(self):
        """
        TODO docstring

        Returns
        -------
        motor_command : (motor_commands_dimensions,) array

        """
        return self.inputs[0][self.input_shape-self.motor_commands_dimensions:].cpu().detach().numpy()

    def get_prediction_horizon(self):
        """
        TODO docstring

        Returns
        -------

        """
        return self.prediction_horizon

    def get_predictions(self, position):
        """
        TODO docstring

        Parameters
        ----------
        position

        Returns
        -------

        """
        return position + torch.cumsum(torch.stack(self.outputs)[:, :position.shape[0]], dim=0)

    def get_action(self, old_state, new_state):
        """
        TODO docstring

        Parameters
        ----------
        old_state : dict
            name : str
            position : (position_dimensions,) array
            velocity : (position_dimensions,) array
            sensor readings : (sensor_readings_dimensions,)
            target : (position_dimensions,) array
        new_state : dict
            name : str
            position : (position_dimensions,) array
            velocity : (position_dimensions,) array
            sensor readings : (sensor_readings_dimensions,)
            target : (position_dimensions,) array

        Returns
        -------
        action : (motor_commands_dimensions,) array
        predictions : (future_time_steps, position_dimensions) array

        """
        self.initialize(old_state, new_state)

        for idx in range(self.inference_iterations):
            self.predict()
            # print('----')
            self.optimize_roll_out()

        self.clean_up(new_state)
        # print('#######')
        print(np.sum(np.square(self.predictions[-1] - new_state['target'])))
        return self.action, self.predictions

    def initialize(self, old_state, new_state):
        """
        TODO docstring

        Parameters
        ----------
        old_state : dict
            name : str
            position : (position_dimensions,) array
            velocity : (position_dimensions,) array
            sensor readings : (sensor_readings_dimensions,)
            target : (position_dimensions,) array
        new_state : dict
            name : str
            position : (position_dimensions,) array
            velocity : (position_dimensions,) array
            sensor readings : (sensor_readings_dimensions,)
            target : (position_dimensions,) array

        Returns
        -------

        """
        self.output_shape = \
            old_state['position'].shape[0] * 2 + \
            old_state['sensor readings'].shape[0]
        self.input_shape = self.output_shape + self.motor_commands_dimensions

        self.mask = torch.zeros(self.input_shape)
        self.mask[self.output_shape:self.input_shape] = torch.ones(self.motor_commands_dimensions)

        self.start_position = torch.tensor(new_state['position'])
        self.target_position = torch.tensor(new_state['target'])

        if self.action is None:
            self.inputs = [
                              torch.zeros(self.input_shape, dtype=torch.float, requires_grad=True)
                          ] * self.prediction_horizon
            self.outputs = [
                               torch.zeros(self.output_shape, dtype=torch.float, requires_grad=True)
                           ] * self.prediction_horizon

        else:
            # self.inputs = util.roll_array_and_fill_last_entry_randomly(self.inputs, 0)
            # print(self.inputs)
            self.inputs = self.inputs[1:] + [torch.rand(self.input_shape, dtype=torch.float, requires_grad=True)]
            # print(self.inputs)
            # self.outputs = util.roll_array_and_fill_last_entry_randomly(self.outputs, 0)
            self.outputs = self.outputs[1:] + [torch.rand(self.output_shape, dtype=torch.float, requires_grad=True)]

        self.position_delta = (new_state['position'] - old_state['position'])

        self.velocity_delta = (new_state['velocity'] - old_state['velocity'])

        self.sensor_readings = new_state['sensor readings']

        self.h_0, self.c_0 = self.model.get_hidden_state()
        self.h_0 = self.h_0.detach()
        self.c_0 = self.c_0.detach()

    def clean_up(self, new_state):
        """
        TODO docstring

        Parameters
        ----------
        new_state

        Returns
        -------

        """
        self.model.set_hidden_state(self.h_0, self.c_0)
        self.action = self.get_next_motor_command()
        self.predictions = self.get_predictions(torch.tensor(new_state['position']).float()).detach().numpy()

    def predict(self):
        """
        TODO docstring

        Returns
        -------

        """
        self.model.set_hidden_state(self.h_0, self.c_0)
        for i in range(self.prediction_horizon):
            self.prepare_input(i)
            self.predict_one_time_step(i)
            # print("in", self.inputs[i])
            # print("out", self.outputs[i])

    def prepare_input(self, i):
        """
        TODO docstring

        Parameters
        ----------
        i

        Returns
        -------

        """
        # print(i)
        if i == 0:
            if self.action is None:
                self.action = -1
                self.inputs[0] = torch.tensor(
                    np.concatenate([self.position_delta,
                                    self.velocity_delta,
                                    self.sensor_readings,
                                    np.random.random(self.motor_commands_dimensions)]),
                    requires_grad=True,
                    dtype=torch.float)
        else:
            self.inputs[i] = torch.cat(
                [
                    self.outputs[i-1],
                    self.inputs[i][self.input_shape-self.motor_commands_dimensions:]
                 ]
            ).clone().detach().requires_grad_(True)

    def predict_one_time_step(self, i):
        """
        TODO docstring

        Parameters
        ----------
        i

        Returns
        -------

        """
        self.outputs[i] = self.model.module(self.inputs[i].reshape(1, 1, -1)).reshape(-1)

    def optimize_roll_out(self):
        """
        TODO docstring

        Returns
        -------

        """
        prediction = self.get_predictions(self.start_position)[-1]
        # prediction = self.start_position
        params = [self.inputs[i] for i in range(self.prediction_horizon)]
        optimizer = torch.optim.Adam(params, lr=self.step_size)
        optimizer.zero_grad()

        position_loss = self.loss(prediction, self.target_position)
        # sensor_readings_loss = self.loss(output_sensor_readings, target_sensor_readings)

        loss = position_loss * self.position_loss_weight  # + sensor_readings_loss * self.sensor_readings_loss_weight
        loss.backward()
        self.mask_gradients(params)
        optimizer.step()
        for i in range(self.prediction_horizon):
            # print(self.inputs[i][self.input_shape-self.motor_commands_dimensions:])
            self.inputs[i].detach_()

    def mask_gradients(self, params):
        """
        TODO docstring

        Parameters
        ----------
        params

        Returns
        -------

        """
        for p in params:
            p.grad *= self.mask
