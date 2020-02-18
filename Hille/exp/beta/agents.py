import active_inference_agent
import data_saver

class Agents:
    """
    wrapper class around all agents
    """
    def __init__(self, agents_config, inference_config, model_config,
                 max_memory_size, num_time_steps, transition_data_type):
        """
        # TODO docstring

        Parameters
        ----------
        agents_config
        inference_config
        model_config
        max_memory_size
        num_time_steps
        transition_data_type
        """
        self.saver = data_saver.Saver()
        self.agents = []
        # self.agents is a list of agent objects that should move in the environment

        for agent_config in agents_config.agents:
            if agent_config.agent_type == 'ActiveInference':
                self.agents.append(
                    active_inference_agent.ActiveInferenceAgent(
                        agent_config,
                        inference_config,
                        model_config,
                        max_memory_size,
                        num_time_steps,
                        transition_data_type
                    )
                )
            elif agent_config.agent_type == 'Obstacle':
                # TODO maybe create own obstacle class
                self.agents.append(
                    active_inference_agent.ActiveInferenceAgent(
                        agent_config,
                        inference_config,
                        model_config,
                        max_memory_size,
                        num_time_steps,
                        transition_data_type
                    )
                )

    def append(self, agent):
        """append one single agent to the list"""
        self.agents.append(agent)

    def reset(self):
        """
        reset every agent and collect initial information

        Returns
        -------
        agents_info : []
            containing dictionaries with
                name : str
                    agent name
                position : (position_dimensions,) array
                mass : float
                radius : float
                target type : str
                prediction horizon : int

        """
        agent_info = []

        for a in self.agents:
            agent_info.append(a.reset())
        print('agent_info', agent_info)
        return agent_info

    def reset_dataset(self):
        for a in self.agents:
            a.reset_dataset()

    def get_train_action(self, state, physic_mode_id, num_thrusters):
        """
        compute actions for every agent

        Parameters
        ----------
        state : []
            containing dictionaries with
                name : str
                position : (position_dimensions,) array
                velocity : (position_dimensions,) array
                sensor readings : (sensor_readings_dimensions,)
                target : (position_dimensions,) array
        physic_mode_id : int
        num_thrusters : int

        Returns
        -------
        actions : []
            containing dictionaries with
                name : str
                motor commands : (motor_commands_dimensions,)
                predictions : (future_time_steps, position_dimensions) array

        """
        actions = []
        names = [s['name'] for s in state]
        for agent in self.agents:
            actions.append(
                agent.get_train_action(state[names.index(agent.name)],
                                       physic_mode_id,
                                       num_thrusters))
        return actions

    def get_inference_action(self, state, physic_mode_id, num_thrusters):
        actions = []
        names = [s['name'] for s in state]
        for agent in self.agents:
            actions.append(agent.get_inference_action(state[names.index(agent.name)],
                                                      physic_mode_id, num_thrusters))
        return actions

    def update(self, time_step, old_state, actions, new_state):
        old_state_names = [s['name'] for s in old_state]
        action_names = [a['name'] for a in actions]
        new_state_names = [s['name'] for s in new_state]
        for agent in self.agents:
            agent.update(time_step,
                         old_state[old_state_names.index(agent.name)],
                         actions[action_names.index(agent.name)],
                         new_state[new_state_names.index(agent.name)])

    def append_transition(self, old_state, actions, new_state):
        """
        save correct state-action-state transition for each agent

        Parameters
        ----------
        old_state : []
            containing dictionaries with
                name : str
                position : (position_dimensions,) array
                velocity : (position_dimensions,) array
                sensor readings : (sensor_readings_dimensions,)
                target : (position_dimensions,) array
        actions : []
            containing dictionaries with
                name : str
                motor commands : (motor_commands_dimensions,)
                predictions : (future_time_steps, position_dimensions) array
        new_state : []
            containing dictionaries with
                name : str
                position : (position_dimensions,) array
                velocity : (position_dimensions,) array
                sensor readings : (sensor_readings_dimensions,)
                target : (position_dimensions,) array

        Returns
        -------

        """
        old_state_names = [s['name'] for s in old_state]
        action_names = [a['name'] for a in actions]
        new_state_names = [s['name'] for s in new_state]
        for agent in self.agents:
            agent.append_transition(old_state[old_state_names.index(agent.name)],
                                    actions[action_names.index(agent.name)],
                                    new_state[new_state_names.index(agent.name)])

    def learn(self):
        """perform one learn step for each agent"""
        for a in self.agents:
            a.learn()

    def save(self, epoch, mini_epoch):
        for a in self.agents:
            self.saver.save_agent(a, epoch, mini_epoch)