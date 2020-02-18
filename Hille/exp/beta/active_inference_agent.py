import numpy as np

import buffer
import dataset
import inference
import memory
import model
import util


class ActiveInferenceAgent:
    def __init__(self, agent_config,
                 inference_config,
                 model_config,
                 max_memory_size,
                 sequence_length,
                 transition_data_type):
        """
        agent that learns the dynamics of an environment through random (or better chosen) actions

        later the learned model can be used in active inference for planning

        Parameters
        ----------
        agent_config : AgentConfig
        inference_config : InferenceConfig
        model_config : ModelConfig
        max_memory_size : int
            maximum size of the ringbuffer (doesnt have to be multiple of batch size)
        sequence_length : int
            length of sequences for which the connection between hidden states is relevant
        transition_data_type : data_type
            similarly to reinforcement learning (s_t, a_t, s_{t+1})

        """
        self.name = agent_config.name
        self.initial_position = agent_config.initial_position
        self.mass = agent_config.mass
        self.radius = agent_config.radius

        assert agent_config.agent_type in ['ActiveInference', 'Obstacle'], \
            'invalid agent_type'
        self.agent_type = agent_config.agent_type

        if not isinstance(agent_config.target_type, list):
            assert agent_config.target_type in [None, 'Random', 'Line',
                                                'Other', 'TargetOfOther',
                                                'Proximity', 'NoProximity'], \
                'invalid target type'
        self.target_type = agent_config.target_type

        # TODO hand over save file name to model
        # TODO maybe create load flag too
        self.save_file = agent_config.save_file

        self.motor_commands_dimensions = transition_data_type['motor commands'].shape[0]
        self.position_dimensions = transition_data_type['old position'].shape[0]

        # version 0
        # self.transition_data_type = transition_data_type
        # self.sequence_buffer = np.empty(self.sequence_length, dtype=self.transition_data_type)
        # self.sequence_buffer_id = 0

        # memory_dtype = np.dtype((self.transition_data_type, (self.sequence_length, )))
        # print(memory_dtype)
        # self.memory = memory.Memory(max_memory_size, memory_dtype, True)

        # version 1
        self.sequence_position = 0
        self.sequence_length = sequence_length
        self.dataset = dataset.Dataset()

        # version 2
        shapes = [[1, ],
                  list(transition_data_type['old position'].shape),
                  list(transition_data_type['old velocity'].shape),
                  list(transition_data_type['old sensor readings'].shape),
                  list(transition_data_type['motor commands'].shape),
                  [model_config.hidden_dim], [model_config.hidden_dim]]
        buffer_keys = ['time step',
                       'position', 'velocity', 'sensor readings', 'motor commands',
                       'hidden state h0', 'hidden state c0']
        self.buffer = buffer.RingBuffer(max_memory_size, sequence_length, shapes, buffer_keys)

        self.model = model.Model(model_config)

        self.last_motor_commands = None
        self.old_state = None

        self.inference = inference.Inference(inference_config, self.model, self.motor_commands_dimensions)

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
        # self.model.reset()
        # self.memory.clear()
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

    def reset_dataset(self):
        self.dataset.reset()

    def append_transition(self, old_state, action, new_state):
        """
        save one state-action-state transition into the memory of this agent

        Parameters
        ----------
        old_state : dict
            name : str
            position : (position_dimensions,) array
            velocity : (position_dimensions,) array
            sensor readings : (sensor_readings_dimensions,)
            target : (position_dimensions,) array
        action : dict
            name : str
            motor commands : (motor_commands_dimensions,)
            predictions : (future_time_steps, position_dimensions) array
        new_state : dict
            name : str
            position : (position_dimensions,) array
            velocity : (position_dimensions,) array
            sensor readings : (sensor_readings_dimensions,)
            target : (position_dimensions,) array

        Returns
        -------

        """
        '''
        # version 0
        transition = np.array(
            [
                (old_state['position'], old_state['velocity'], old_state['sensor readings'],
                 action['motor commands'],
                 new_state['position'], new_state['velocity'], new_state['sensor readings'])
            ], dtype=self.transition_data_type
        )

        self.sequence_buffer[self.sequence_buffer_id] = transition
        self.sequence_buffer_id += 1
        if self.sequence_buffer_id == len(self.sequence_buffer):
            self.append_sequence(self.sequence_buffer)
            self.sequence_buffer = np.empty(self.sequence_length, self.transition_data_type)
            self.sequence_buffer_id = 0
        '''
        # version 1
        if self.sequence_position % self.sequence_length == 0:
            self.dataset.start_new_series()
        self.dataset.append([old_state['position'], old_state['velocity'], old_state['sensor readings'],
                             action['motor commands'],
                             new_state['position'], new_state['velocity'], new_state['sensor readings']])
        self.sequence_position += 1
        if self.sequence_position % self.sequence_length == 0:
            self.dataset.transform_old_series()

    def update(self, time_step, old_state, action, new_state):
        # TODO rework !?
        self.append_to_buffer(time_step, old_state, action)
        # self.model.refresh(self.buffer)

    def append_to_buffer(self, time_step, old_state, action):
        # version 2
        h0, c0 = self.model.get_hidden_state()

        # for 'true' model current hidden state is only interesting therefor take first batch element
        h0, c0 = h0[0].detach().numpy(), c0[0].detach().numpy()

        item = {'time step': time_step,
                'position': old_state['position'],
                'velocity': old_state['velocity'],
                'sensor readings': old_state['sensor readings'],
                'motor commands': action['motor commands'],
                'hidden state h0': h0,
                'hidden state c0': c0}
        self.buffer.append(item)

    def append_sequence(self, sequence):
        """append one sequence to the ringbuffer """
        # self.memory.append(sequence)
        raise NotImplementedError

    # unused
    def extend_sequences(self, sequences):
        """extend the ringbuffer with several sequences
        (with array like data structure that numpy can understand)"""
        # self.memory.extend(sequences)
        raise NotImplementedError

    def learn(self):
        """improve the model"""
        # self.model.learn(self.memory)
        # self.model.learn_dataset(self.dataset)
        self.model.learn_buffer(self.buffer)

    def get_train_action(self, state, physic_mode_id, num_thrusters):
        """
        compute an actions that the agent should perform in the given state

        while still learning the model

        Parameters
        ----------
        state : dict
            name : str
            position : (position_dimensions,) array
            velocity : (position_dimensions,) array
            sensor readings : (sensor_readings_dimensions,)
            target : (position_dimensions,) array
        physic_mode_id : int
        num_thrusters : int

        Returns
        -------
        actions : dict
            name : str
            motor commands : (motor_commands_dimensions,) array
            predictions : (future_time_steps, position_dimensions) array

        """
        # ucb and stuff
        # TODO this is my actual task, not reworking this framework...
        if self.last_motor_commands is None:
            self.last_motor_commands = np.random.random(4)
        self.last_motor_commands = \
            util.get_random_motor_commands(self.last_motor_commands,
                                           physic_mode_id,
                                           num_thrusters)
        # TODO create motor commands for prediction horizon and
        # TODO get predictions as well
        return {'name': state['name'],
                'motor commands': self.last_motor_commands,
                'predictions': np.zeros((self.inference.get_prediction_horizon(),
                                         self.position_dimensions))}

    def reset_inference(self):
        self.inference.reset()

    def get_inference_action(self, new_state, physic_mode_id, num_thrusters):
        """
        perform action inference starting at position to reach target

        Parameters
        ----------
        new_state : dict
            name : str
            position : (position_dimensions,) array
            velocity : (position_dimensions,) array
            sensor readings : (sensor_readings_dimensions,)
            target : (position_dimensions,) array

        Returns
        -------
        actions : dict
            name : str
            motor commands : (motor_commands_dimensions,) array
            predictions : (future_time_steps, position_dimensions) array

        """
        if self.buffer.is_full():
            if self.old_state is None:
                self.old_state = new_state
            new_motor_commands, predictions = self.inference.get_action(self.old_state, new_state)
            self.old_state = new_state
            return {'name': new_state['name'],
                    'motor commands': new_motor_commands,
                    'predictions': predictions}
        else:
            return self.get_train_action(new_state, physic_mode_id, num_thrusters)
