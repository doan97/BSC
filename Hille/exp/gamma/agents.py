import logging

import numpy as np

import buffer
import inference
import model


class BaseAgent:
    def __init__(self):
        """
        TODO docstring
        """
        pass

    def reset(self):
        """
        TODO docstring

        Returns
        -------

        """
        raise NotImplementedError

    def act(self, *args):
        """
        TODO docstring

        Parameters
        ----------
        args

        Returns
        -------

        """
        raise NotImplementedError

    def learn(self, *args):
        """
        TODO docstring

        Parameters
        ----------
        args

        Returns
        -------

        """
        print(' noooooooooooooooooooooooooo ')
        raise NotImplementedError


class ActiveInferenceAgent(BaseAgent):
    def __init__(self, name, initial_position, mass, radius, inference_config, model_config, action_space, state_space,
                 target_type, seed=0):
        """
        agent that learns the dynamics of an environment through random (or better chosen) actions

        later the learned model can be used in active inference for planning

        Parameters
        ----------
        name : str
        initial_position : iter
        mass : float
        radius : float
        inference_config : InferenceConfig
        model_config : ModelConfig
        action_space : int
        state_space : iter
            position, velocity, sensor

        """
        super().__init__()
        self.name = name
        self.initial_position = np.array(initial_position)
        self.mass = mass
        self.radius = radius

        if target_type not in [None, 'Random', 'Line', 'Other', 'TargetOfOther', 'Proximity', 'NoProximity']:
            logging.error('invalid target type')
        self.target_type = target_type

        self.motor_commands_dimensions = action_space
        self.position_dimensions = state_space[0]
        self.sensor_reading_dimensions = state_space[2]

        model_config['output dimension'] = state_space[0] + state_space[1] + state_space[2]
        model_config['input dimension'] = model_config['output dimension'] + action_space

        # version 1
        self.sequence_position = 0
        self.sequence_length = model_config['sequence length']

        # version 2
        shapes = [[1, ],
                  [self.position_dimensions, ],
                  [self.position_dimensions, ],
                  [self.sensor_reading_dimensions, ],
                  [self.motor_commands_dimensions, ],
                  [model_config['hidden dimension']],
                  [model_config['hidden dimension']]]
        keys = ['time step',
                'position', 'velocity', 'sensor readings', 'motor commands',
                'hidden state h0', 'hidden state c0']
        self.buffer = buffer.RingBuffer(model_config['maximum memory size'], self.sequence_length, shapes, keys)

        self.model = model.Model(model_config, seed)

        self.last_motor_commands = None
        self.old_state = None

        self.inference = inference.Inference(inference_config, self.model, self.motor_commands_dimensions, seed)

    def reset(self):
        """
        reset the agent for new epoch

        Returns
        -------
        agent_info : dict
            name : str
            position : (position_dimensions,) array
            mass : float
            radius : float
            target type : str
            prediction horizon : int

        """
        self.buffer.reset()
        self.model.reset()
        self.inference.reset()
        agent_info = {
            'name': self.name,
            'position': self.initial_position,
            'mass': self.mass,
            'radius': self.radius,
            'target type': self.target_type,
            'prediction horizon': self.inference.get_prediction_horizon()
        }

        # TODO check if there is some stuff missing
        self.old_state = None
        return agent_info

    def act(self, new_state, physic_mode_id, num_thrusters):
        """
        TODO docstring

        Parameters
        ----------
        new_state
        physic_mode_id
        num_thrusters

        Returns
        -------

        """
        if self.old_state is None:
            self.old_state = new_state
        new_motor_commands, predictions = self.inference.get_action(self.old_state, new_state)
        self.old_state = new_state
        return {'name': new_state['name'],
                'motor commands': new_motor_commands,
                'predictions': predictions}

    def update(self, time_step, state, action):
        """
        TODO docstring

        Parameters
        ----------
        time_step : int
        state : dict
            name : str
            position : (position_dimensions,) array
            velocity : (position_dimensions,) array
            sensor readings : (sensor_readings_dimensions,)
            target : (position_dimensions,) array
        action : dict
            name : str
            motor commands : (motor_commands_dimensions,)
            predictions : (future_time_steps, position_dimensions) array

        Returns
        -------

        """
        h0, c0 = self.model.get_hidden_state()

        # for 'true' model current hidden state is only interesting therefor take first batch element
        h0, c0 = h0[0].detach().numpy(), c0[0].detach().numpy()

        item = {'time step': time_step,
                'position': state['position'],
                'velocity': state['velocity'],
                'sensor readings': state['sensor readings'],
                'motor commands': action['motor commands'],
                'hidden state h0': h0,
                'hidden state c0': c0}
        self.buffer.append(item)

    def learn(self):
        """improve the model"""
        self.model.learn(self.buffer)
