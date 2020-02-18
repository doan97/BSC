

class Handler:
    """
    class that organises step and learn for all agents
    """
    def __init__(self):
        """
        TODO docstring

        """
        # self.agents is a list of agent objects that should move in the environment
        self.agents = []

    def append(self, new_agent):
        """append one single agent to the list"""
        self.agents.append(new_agent)

    def reset(self):
        """
        reset every agent and collect initial information

        the memory is cleared as well

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

    def act(self, state=None, physic_mode_id=None, num_thrusters=None):
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
                agent.act(state[names.index(agent.name)], physic_mode_id,  num_thrusters))
        return actions

    def update(self, time_step, state, actions):
        """
        TODO docstring

        Parameters
        ----------
        time_step : int
        state : []
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

        Returns
        -------

        """
        state_names = [s['name'] for s in state]
        action_names = [a['name'] for a in actions]
        for agent in self.agents:
            agent.update(time_step,
                         state[state_names.index(agent.name)],
                         actions[action_names.index(agent.name)])

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
