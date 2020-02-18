import numpy as np

import gui
import targets
import util
#from data_saver import Saver


class Environment:
    def __init__(self, env_config, gui_config, stopwatch=None, seed=0):
        """
        create simulator for the physical dynamics

        usage based on open-ai gym api (https://github.com/openai/gym)

            import gym
            from gym import target_error, spaces, utils
            from gym.utils import seeding

            class FooEnv(gym.Env):
              metadata = {'render.modes': ['human']}

              def __init__(self):
                ...
              def step(self, actions):
                ...
              def reset(self):
                ...
              def render(self, mode='human'):
                ...
              def close(self):
                ...

        additionally the following methods are 'public'
            change_physic_mode()
            change_obstacle_transformation()

        Parameters
        ----------
        env_config
        stopwatch
        """
        self.verbose = env_config.verbose

        self.time_step = 0

        self.stopwatch = stopwatch
        self.seed = np.random.seed(seed)

        self.physic_mode_id = -1
        self.physic_mode_names = env_config.physic_mode_names

        self.max_thrust = env_config.max_thrust
        self.left_border = env_config.left_border
        self.right_border = env_config.right_border
        self.top_border = env_config.top_border
        self.bot_border = env_config.bot_border
        self.borders = np.array([
            [[self.left_border, self.bot_border],
             [self.left_border, self.top_border]],
            [[self.right_border, self.bot_border],
             [self.right_border, self.top_border]],
            [[self.left_border, self.top_border],
             [self.right_border, self.top_border]],
            [[self.left_border, self.bot_border],
             [self.right_border, self.bot_border]]
        ])
        self.delta_time = env_config.delta_time

        self.number_of_problems = env_config.number_of_problems
        self.min_motor_value = env_config.min_motor_value
        self.max_motor_value = env_config.max_motor_value

        self.point_spread_flag = env_config.point_spread_flag
        self.point_spread_size = env_config.point_spread_size
        self.point_spread_type = env_config.point_spread_type
        self.point_spread_sigma = env_config.point_spread_sigma

        self.point_spread_clip_min = env_config.point_spread_clip_min
        self.point_spread_clip_max = env_config.point_spread_clip_max
        self.point_spread_round_decimals = \
            env_config.point_spread_round_decimals

        self.normalize_sensor_readings = env_config.normalize_sensor_readings

        self.max_distance = env_config.max_distance
        self.border_proximity_weight = env_config.border_proximity_weight

        self.position_dimensions = env_config.position_dimensions
        self.sensor_readings_dimensions = env_config.sensor_readings_dimensions
        self.motor_commands_dimensions = env_config.motor_commands_dimensions

        self.g = None

        self.floor_friction = None
        self.ceilingFriction = None
        self.side_friction = None
        self.agent_friction = None

        self.numThrusts = None

        self.sensor_directions = \
            util.calc_sensor_directions(self.sensor_readings_dimensions)

        self.thrustDirections = util.calc_thrust_directions()
        self.tmp_counter = 0

        self.number_of_obstacles = env_config.number_of_obstacles

        if env_config.positions_of_obstacles is not None:
            assert \
                len(env_config.positions_of_obstacles) \
                == self.number_of_obstacles, \
                'please provide exactly one position for each obstacle ' \
                'or set None'

        self.positions_of_obstacles = env_config.positions_of_obstacles

        if isinstance(env_config.radii_of_obstacles, list):
            assert \
                len(env_config.radii_of_obstacles) \
                == self.number_of_obstacles, \
                'please provide exactly one radius for each obstacle ' \
                'or only one float value'

        self.radii_of_obstacles = env_config.radii_of_obstacles

        self.obstacle_transformation_id = -1
        self.obstacle_transformation = None
        self.obstacle_transformation_names = \
            env_config.obstacle_transformation_names

        self.max_obstacle_start_velocity = \
            env_config.max_absolute_obstacle_start_velocity
        self.obstacle_line_acceleration_factor = \
            env_config.obstacle_line_acceleration_factor

        self.max_absolute_obstacle_rotation_radius = \
            env_config.max_absolute_obstacle_rotation_radius
        self.obstacle_curve_acceleration_factor = \
            env_config.obstacle_curve_acceleration_factor

        self.default_object_mass = env_config.default_object_mass

        self.env_objects = []
        """
        env_objects : list
            containing dictionaries with
                name : str
                type : str
                position': (position_dimensions,) array
                velocity': (position_dimensions,) array
                mass : float
                radius : float
                color : str
                (and then depending on type)
                if type=='agent':
                    thruster activity : (motor_commands_dimensions,) array
                    sensor readings : (sensor_readings_dimensions,) array
                    target_type : str
                    predictions : 
                        (prediction_horizon, position_dimensions) array
                    target : (position_dimensions,) array
                    target iterator : iterator from targets.py
                elif type=='obstacle':
                    angle : float
                    delta : float
                    rotation radius : float
                
        """
        if self.position_dimensions == 2:
            self.gui_active = True
            self.gui = gui.GUI(gui_config,
                               env_config.size,
                               env_config.sensor_readings_dimensions)
        else:
            print('cant use gui for {} position dimensions'.format(
                self.position_dimensions))
            self.gui_active = False

    def reset(self, agents_info):
        """
        reset the environment to the default start state

        mostly used for a new epoch

        this method should be called before step
        (or else the next state is not well defined)

        Parameters
        ----------
        agents_info : []
            containing dictionaries with
                name : str
                position : (position_dimensions,) array
                mass : float
                radius : float
                target type : str
                prediction horizon : int

        Returns
        -------
        state : []
            containing dictionaries with
                name : str
                position : (position_dimensions,) array
                velocity : (position_dimensions,) array
                sensor readings : (sensor_readings_dimensions,)
                target : (position_dimensions,) array

        """
        self.time_step = 0
        self.env_objects = []
        self.create_objects(agents_info)
        self.create_targets()
        self.update_targets()
        self.update_sensor_readings()

        if self.verbose:
            print('current env objects:')
            for env_object in self.env_objects:
                print(env_object)

        if self.gui_active:
            self.gui.reset(self.build_env_info())

        return self.build_state()

    def create_objects(self, agents_info):
        """fill the environment with all necessary objects"""
        self.create_agents(agents_info)
        self.create_obstacles()

    def create_obstacles(self):
        """
        fill the environment with obstacles

        if positions or radii are specified in config,
        then there should be as many as number
        """
        for idx in range(self.number_of_obstacles):
            if isinstance(self.radii_of_obstacles, list):
                r = self.radii_of_obstacles[idx]
            else:
                r = self.radii_of_obstacles
            if self.positions_of_obstacles is None:
                low = [self.left_border + r, self.bot_border + r]
                high = [self.right_border - r, self.top_border - r]
                pos = np.random.uniform(low, high)
            else:
                pos = self.positions_of_obstacles[idx]
            obstacle = {'name': 'obstacle_{}'.format(idx),
                        'type': 'obstacle',
                        'position': pos,
                        'velocity':
                            (np.random.random(2)-0.5)
                            * self.max_obstacle_start_velocity,
                        'mass': self.default_object_mass,
                        'radius': r,
                        'angle': np.random.rand() * np.pi * 2,
                        'delta': np.random.rand() * np.pi / 50,
                        'rotation radius':
                            (np.random.rand() + 0.5)
                            * self.max_absolute_obstacle_rotation_radius}
            self.env_objects.append(obstacle)

    def create_agents(self, agents_info):
        """
        fill environment with several agents

        Parameters
        ----------
        agents_info : []
            containing dictionaries with
                name : str
                position : (position_dimensions,) array
                mass : float
                radius : float
                target type : str
                prediction horizon : int

        Returns
        -------

        """
        for agent_info in agents_info:
            self.create_agent(agent_info)

    def create_agent(self, agent_info):
        """fill environment with one agent

        Parameters
        ----------
        agent_info : dict
            containing
                name : str
                position : (position_dimensions,) array
                mass : float
                radius : float
                target type : str
                prediction horizon : int

        Returns
        -------

        """
        self.env_objects.append({
            'name': agent_info['name'],
            'type': 'agent',
            'position': agent_info['position'],
            'velocity': np.zeros(self.position_dimensions),
            'mass': agent_info['mass'],
            'radius': agent_info['radius'],

            'thruster activity': np.zeros(self.motor_commands_dimensions),
            'sensor readings': np.zeros(self.sensor_readings_dimensions),

            'target type': agent_info['target type'],
            'predictions': np.zeros((agent_info['prediction horizon'],
                                     self.position_dimensions)),
            'target': np.zeros(self.position_dimensions)
        })

    def create_targets(self):
        """create targets for all agents in the environment"""
        for env_object in self.env_objects:
            if env_object['type'] == 'agent':
                self.create_target_iterator_for_one_agent(env_object)

    def create_target_iterator_for_one_agent(self, agent_object,
                                             target_lifetime=1000, target_refresh_rate=100):
        """
        construct target iterator for the current agent

        because targets truly change only with reset of environment,
        the life time and refresh rate are merely included for completeness sake

        Parameters
        ----------
        agent_object : dict
            name : str
            type : str
            position': (position_dimensions,) array
            velocity': (position_dimensions,) array
            mass : float
            radius : float
            color : str
            thruster activity : (motor_commands_dimensions,) array
            sensor readings : (sensor_readings_dimensions,) array
            target type : str
            predictions : (prediction_horizon, position_dimensions) array
            target : (position_dimensions,) array
            target iterator : iterator from targets.py
        target_lifetime : int
            how long target iterator can be used
        target_refresh_rate : int
            how long before target is refreshed (handled by iterator)

        Returns
        -------

        """
        if agent_object['target type'] == 'Random':
            low = [self.left_border + agent_object['radius'],
                   self.bot_border + agent_object['radius']]
            high = [self.right_border - agent_object['radius'],
                    self.top_border - agent_object['radius']]
            agent_object['target iterator'] = targets.RandomTarget(None, target_lifetime,
                                                                   target_refresh_rate, low, high)

        elif agent_object['target type'] == 'TargetOfOther':
            other_agent = None
            for other_object in self.env_objects:
                if other_object['type'] == 'agent':
                    if other_object['name'] != agent_object['name']:
                        if other_agent is None:
                            other_agent = other_object
                        else:
                            raise RuntimeError('target of other agent is not well defined '
                                               '(probably more than one)')
            agent_object['target iterator'] = targets.OtherTarget(None, target_lifetime,
                                                                  other_agent)

        elif agent_object['target type'] == 'Other':
            other_agent = None
            for other_object in self.env_objects:
                if other_object['type'] == 'agent':
                    if other_object['name'] != agent_object['name']:
                        if other_agent is None:
                            other_agent = other_object
                        else:
                            raise RuntimeError('other agent is not well defined '
                                               '(probably more than one)')
            agent_object['target iterator'] = targets.AgentTarget(None, target_lifetime,
                                                                  other_agent)

        elif agent_object['target type'] == 'Line':
            low = [self.left_border + agent_object['radius'],
                   self.bot_border + agent_object['radius']]
            high = [self.right_border - agent_object['radius'],
                    self.top_border - agent_object['radius']]
            agent_object['target iterator'] = targets.LineTarget(None, target_lifetime,
                                                                 target_refresh_rate, low, high)

        elif agent_object['target type'] == 'Proximity':
            # TODO implement this in targets.py if this is even possible there
            agent_object['target iterator'] = targets.ProximityTarget(None, target_lifetime,
                                                                      target_refresh_rate, None, None, True)

        elif agent_object['target type'] == 'NoProximity':
            # TODO implement this in targets.py if this is even possible there
            agent_object['target iterator'] = targets.ProximityTarget(None, target_lifetime,
                                                                      target_refresh_rate, None, None, False)

        elif isinstance(agent_object['target type'], (np.ndarray, list)):
            assert len(agent_object['target type']) == self.position_dimensions, \
                'invalid target shape {}'.format(len(agent_object['target type']))
            agent_object['target iterator'] = targets.ConstantTarget(np.array(agent_object['target type']),
                                                                     target_lifetime)

        elif agent_object['target type'] is None:
            agent_object['target iterator'] = targets.ConstantTarget(np.array(agent_object['position']),
                                                                     target_lifetime)

        else:
            raise NotImplementedError('target type {} for agent {} not known'
                                      .format(agent_object['target type'], agent_object['name']))

    def build_env_info(self):
        """
        collect necessary info for gui

        contains every agent, its predictions and target,
        and all obstacles

        Returns
        -------
        env_objects_info : []
            containing dictionaries with
                name : str
                type : str
                position : (position_dimensions,) array
                velocity : (position_dimensions,) array
                radius : float
                if type == 'agent':
                    thrust number : int
                    thruster activity : (motor_commands_dimensions,) array
                    predictions :
                        (prediction_horizon, position_dimensions) array
                    target : (position_dimensions,) array

        """
        env_objects_info = []

        for env_object in self.env_objects:
            info = {
                'name': env_object['name'],
                'type': env_object['type'],
                'position': env_object['position'],
                'velocity': env_object['velocity'],
                'radius': env_object['radius']

            }
            if env_object['type'] == 'agent':
                info['thrust number'] = self.numThrusts
                info['thruster activity'] = env_object['thruster activity']
                if env_object['target type'] is None:
                    info['predictions'] = np.stack([env_object['position']] *
                                                   env_object['predictions'].shape[0])
                else:
                    info['predictions'] = env_object['predictions']
                info['target'] = env_object['target']

            env_objects_info.append(info)

        return env_objects_info

    def build_texts_info(self):
        """collect information to display as text"""
        return {
            'time step': self.time_step,
            # TODO include prediction and target error somehow...
            'prediction error': np.random.rand(),
            'target error': np.random.rand()
        }

    def build_state(self):
        """
        collect accessible state info for all agents

        Returns
        -------
        state : []
            containing dictionaries with
                name : str
                position : (position_dimensions,) array
                velocity : (position_dimensions,) array
                sensor readings : (sensor_readings_dimensions,)
                target : (position_dimensions,) array

        """
        state = []
        for env_object in self.env_objects:
            if env_object['type'] == 'agent':
                state.append({
                    'name': env_object['name'],
                    'position': env_object['position'],
                    'velocity': env_object['velocity'],
                    'sensor readings': env_object['sensor readings'],
                    'target': env_object['target']
                })
        return state

    def render(self, arguments=None):
        """
        visualize the current state of the environment

        Parameters
        ----------
        arguments : dict
            dictionary with render arguments

        Returns
        -------

        """
        if arguments is not None:
            raise NotImplementedError

        if self.gui_active:
            self.gui.update(self.build_env_info(),
                            self.build_texts_info())

    def close(self):
        """
        perform cleanup stuff

        including shutting down and closing rendered windows

        Returns
        -------

        """
        if self.gui_active:
            self.gui.close()

    def seed(self, seed):
        """set seed for random number generators that the environment uses"""
        self.seed = np.random.seed(seed)

    def get_physic_mode_id(self):
        """{insert trivial documentation here}"""
        return self.physic_mode_id

    def set_physic_mode_id(self, idx):
        """{insert trivial documentation here}"""
        self.physic_mode_id = idx

    def get_physic_mode_name(self):
        """get the current physic mode name"""
        return self.physic_mode_names[self.physic_mode_id]

    def change_physic_mode(self, physic_mode=None):
        """change physic mode via name if one is given"""
        if physic_mode is None:
            self.physic_mode_id = \
                (self.get_physic_mode_id() + 1) % self.number_of_problems
        elif physic_mode in self.physic_mode_names:
            self.physic_mode_id = self.physic_mode_names.index(physic_mode)
        elif physic_mode == 'Random':
            self.physic_mode_id = \
                np.random.randint(len(self.physic_mode_names))
        else:
            raise NotImplementedError
        self.set_physic_mode()

    def set_physic_mode(self):
        """set physic mode of the environment"""
        if self.physic_mode_id == 0:
            """Rocket"""
            self.g = 9.81

            self.floor_friction = .9
            self.ceilingFriction = .9
            self.side_friction = .9
            self.agent_friction = .9

            self.numThrusts = 2

        elif self.physic_mode_id == 1:
            """Glider"""
            self.g = 0.0

            self.floor_friction = .9
            self.ceilingFriction = .9
            self.side_friction = .9
            self.agent_friction = .4

            self.numThrusts = 4

        elif self.physic_mode_id == 2:
            """Stepper"""
            self.g = 0.0

            self.floor_friction = 1
            self.ceilingFriction = 1
            self.side_friction = 1
            self.agent_friction = 1

            self.numThrusts = 4

        else:
            raise SystemExit("SIMULATOR MODE UNKNOWN")
        if self.verbose:
            print('set physic mode to {}'.format(
                self.physic_mode_names[self.physic_mode_id]))

    def get_obstacle_transformation_id(self):
        """{insert trivial documentation here}"""
        return self.obstacle_transformation_id

    def set_obstacle_transformation_id(self, idx):
        """{insert trivial documentation here}"""
        self.obstacle_transformation_id = idx

    def get_obstacle_transformation_name(self):
        """return the name of the current obstacle transformation"""
        return self.obstacle_transformation_names[
            self.obstacle_transformation_id]

    def change_obstacle_transformation(self, transformation=None):
        """change obstacle transformation via name if one is given"""
        if transformation is None:
            self.obstacle_transformation_id = \
                (self.get_obstacle_transformation_id() + 1) \
                % len(self.obstacle_transformation_names)
        elif transformation in self.obstacle_transformation_names:
            self.obstacle_transformation_id = \
                self.obstacle_transformation_names.index(transformation)
        elif transformation == 'Random':
            self.obstacle_transformation_id = \
                np.random.randint(len(self.obstacle_transformation_names))
        else:
            raise NotImplementedError
        if self.verbose:
            print('set obstacle transformation to {}'.format(
                self.obstacle_transformation_names[
                    self.obstacle_transformation_id]))

    def step(self, actions):
        """
        perform one step in the environment

        Parameters
        ----------
        actions : []
            containing dictionaries with
                name : str
                motor commands : (motor_commands_dimensions,) array
                predictions : (future_time_steps, position_dimensions) array

        Returns
        -------
        state : []
            containing dictionaries with
                name : str
                position : (position_dimensions,) array
                velocity : (position_dimensions,) array
                sensor readings : (sensor_readings_dimensions,)
                target : (position_dimensions,) array

        """
        self.time_step += 1
        self.move_objects(actions)
        self.update_targets()
        self.update_sensor_readings()
        return self.build_state()

    def move_objects(self, actions):
        """
        move all objects in the environment

        Parameters
        ----------
        actions : []
            containing dictionaries with
                name : str
                motor commands : (motor_commands_dimensions,) array
                predictions : (future_time_steps, position_dimensions) array

        Returns
        -------

        """
        for env_object in self.env_objects:
            if env_object['type'] == 'agent':
                self.move_agent(
                    env_object,
                    actions[[action['name'] for action in actions].index(
                        env_object['name'])])
            elif env_object['type'] == 'obstacle':
                self.move_obstacle(env_object)
            else:
                raise NotImplementedError

    def line(self, pos, vel, acc=False):
        """
        calculate next position and next velocity for point in two dimensions

        movement follows straight line

        Parameters
        ----------
        pos : (2,) array
            current position vector
        vel : (2,) array
            current velocity vector
        acc : bool
            whether to accelerate or not

        Returns
        -------

        """
        if acc:
            return pos + vel, vel * self.obstacle_line_acceleration_factor
        else:
            return pos + vel, vel

    def curve(self, pos, angle, delta, rotation_radius, acc=False):
        """
        calculate next position for point in two dimensions

        movement follows curve with rotation radius and angle

        Parameters
        ----------
        pos : (2,) array
            current position vector
        angle : float
            current position angle
        delta : float
            change in position angle
        rotation_radius : float
            current position radius
        acc : bool
            whether to accelerate or not

        Returns
        -------
        pos : (2,) array
        angle : float
        delta : float

        """
        pos[0] -= np.math.cos(angle) * rotation_radius
        pos[1] -= np.math.sin(angle) * rotation_radius
        angle = (angle + delta) % (np.pi * 2)
        pos[0] += np.math.cos(angle) * rotation_radius
        pos[1] += np.math.sin(angle) * rotation_radius

        if acc:
            delta *= self.obstacle_curve_acceleration_factor

        return pos, angle, delta

    def move_obstacle(self, obstacle_object):
        """
        apply transformation for given obstacle

        Parameters
        ----------
        obstacle_object : dict
            name : str
            type : str
            position': (position_dimensions,) array
            velocity': (position_dimensions,) array
            mass : float
            radius : float
            color : str
            angle : float
            delta : float
            rotation radius : float

        Returns
        -------

        """

        if self.obstacle_transformation_id == 0:
            """static"""
            pass

        elif self.obstacle_transformation_id == 1:
            """line"""
            old_position = obstacle_object['position']
            old_velocity = obstacle_object['velocity']
            new_position, new_velocity = self.line(old_position, old_velocity)
            obstacle_object['position'] = new_position
            obstacle_object['velocity'] = new_velocity

        elif self.obstacle_transformation_id == 2:
            """curve"""
            old_position = obstacle_object['position']
            old_angle = obstacle_object['angle']
            old_delta = obstacle_object['delta']
            old_rotation_radius = obstacle_object['rotation radius']
            new_position, new_angle, _ = \
                self.curve(old_position, old_angle,
                           old_delta, old_rotation_radius)
            obstacle_object['position'] = new_position
            obstacle_object['angle'] = new_angle

        elif self.obstacle_transformation_id == 3:
            """line accelerated"""
            old_position = obstacle_object['position']
            old_velocity = obstacle_object['velocity']
            new_position, new_velocity = \
                self.line(old_position, old_velocity, acc=True)
            obstacle_object['position'] = new_position
            obstacle_object['velocity'] = new_velocity

        elif self.obstacle_transformation_id == 4:
            """curve accelerated"""
            old_position = obstacle_object['position']
            old_angle = obstacle_object['angle']
            old_delta = obstacle_object['delta']
            old_rotation_radius = obstacle_object['rotation radius']
            new_position, new_angle, new_delta = \
                self.curve(old_position, old_angle,
                           old_delta, old_rotation_radius, acc=True)
            obstacle_object['position'] = new_position
            obstacle_object['angle'] = new_angle
            obstacle_object['delta'] = new_delta

        else:
            raise NotImplementedError

    def move_agent(self, agent_object, action):
        """
        move one agent according to the given actions

        Parameters
        ----------
        agent_object : dict
            name : str
            type : str
            position': (position_dimensions,) array
            velocity': (position_dimensions,) array
            mass : float
            radius : float
            color : str
            thruster activity : (motor_commands_dimensions,) array
            sensor readings : (sensor_readings_dimensions,) array
            target_type : str
            predictions : (prediction_horizon, position_dimensions) array
            target : (position_dimensions,) array
        action : (motor_commands_dimensions,) array
            motor commands that the current agent chose as control signals

        Returns
        -------

        """
        # if self.verbose:
        #    print('now in move agent {} with action {}'.format(
        #        agent_object['name'], action))
        if agent_object['target type'] is None:
            return
        mass = agent_object['mass']
        gravity = np.array([0.0, -1.0]) * mass * self.g
        agent_object['thruster activity'] = \
            self.get_thrusts_from_motor_commands(action['motor commands'])

        # calculate the force that is applied in x an y direction
        # orientation of thrusters is important
        thrust_forces = np.matmul((agent_object['thruster activity']
                                   * self.max_thrust)[np.newaxis, :],
                                  self.thrustDirections)

        # take gravity into account
        force_sum = thrust_forces[0] + gravity

        # a = F/m
        acceleration = force_sum / mass

        # consequent velocity without borders.
        # v = delta_s / delta_t
        # or
        # delta_v = a * delta_t
        # v = v_0 + delta_v
        velocity_change = acceleration * self.delta_time
        hyp_vel = agent_object['velocity'] + velocity_change

        # consequent position without borders
        # delta_s = v * delta_t
        # s = s_0 + delta_s
        position_change = hyp_vel * self.delta_time
        hyp_pos = agent_object['position'] + position_change

        # Calculate velocity and position
        # after consideration of borders and other agents
        new_velocity, new_position = \
            self.check_borders(hyp_vel, hyp_pos,
                               agent_object['radius'],
                               current_agent_name=agent_object['name'])

        if self.physic_mode_id == 2 or self.physic_mode_id == 3:
            new_velocity = np.zeros(self.position_dimensions)

        # ================================
        # Now calculate the real Position
        # ================================
        # In delta-Mode, the Simulator
        # and the NN calculate the difference in position
        # between the previous and the current time step.
        # The Input for the next NN-iteration
        # however must be the real new position and not the delta.
        # That's why sensorOutput and nextSensorInput are separated

        agent_object['position'] = new_position
        agent_object['velocity'] = new_velocity

        agent_object['predictions'] = action['predictions']

        # TODO find original use for position_delta, acceleration
        # TODO adapt this for active inference
        #  that means no hard changes without flag or something

    def get_thrusts_from_motor_commands(self, motor_commands):
        """
        transform motor commands into thruster activation

        Parameters
        ----------
        motor_commands : (motor_commands_dimensions,) array

        Returns
        -------
        thrusts : (motor_commands_dimensions,) array

        """
        motor_commands = \
            np.clip(motor_commands, self.min_motor_value, self.max_motor_value)

        thrusts = np.zeros(self.motor_commands_dimensions)
        thrusts[0:2] = motor_commands[0:2]

        if self.physic_mode_id == 1 or self.physic_mode_id == 2:
            thrusts[2:4] = motor_commands[2:4]

        # TODO get info on why the mode 4 stepper2dir is depreciated
        # else:
        # only for Stepper2Dir
        # changeInOrientation = (motorCommands[3] - motorCommands[2]) * np.pi
        return thrusts

    def check_borders(self, hyp_vel, hyp_pos, radius, current_agent_name):
        """
        check for collisions with borders and other agents

        returns the updated velocities and positions

        Parameters
        ----------
        hyp_vel : (position_dimensions,) array
        hyp_pos : (position_dimensions,) array
        radius : float
        current_agent_name : str

        Returns
        -------
        hyp_vel : (position_dimensions,) array
        hyp_pos : (position_dimensions,) array

        """
        # TODO maybe rework check methods for simpler usage
        hyp_vel, hyp_pos = self.check_y_borders(hyp_vel, hyp_pos, radius)
        hyp_vel, hyp_pos = self.check_x_borders(hyp_vel, hyp_pos, radius)
        hyp_vel, hyp_pos = \
            self.check_agents(hyp_vel, hyp_pos, radius, current_agent_name)

        return hyp_vel, hyp_pos

    def check_y_borders(self, hyp_vel, hyp_pos, radius):
        """
        check for collisions with top or bottom border

        Parameters
        ----------
        hyp_vel : (position_dimensions,) array
        hyp_pos : (position_dimensions,) array
        radius : float

        Returns
        -------
        new_velocity : (position_dimensions,) array
        new_positions : (position_dimensions,) array

        """
        # new velocities and position without border
        new_velocity = np.copy(hyp_vel)
        new_position = np.copy(hyp_pos)

        # -----
        # differences between bottom and top border
        # -----
        bottom_diff = hyp_pos[1] - (self.bot_border + radius)
        top_diff = hyp_pos[1] - (self.top_border - radius)

        # check borders
        # too far bot
        if bottom_diff <= 0:
            new_velocity[0] = hyp_vel[0] * self.floor_friction
            new_velocity[1] = 0.0
            new_position[1] = self.bot_border + radius

        # too far top
        elif top_diff >= 0:
            new_velocity[0] = hyp_vel[0] * self.ceilingFriction
            new_velocity[1] = 0.0
            new_position[1] = self.top_border - radius

        return new_velocity, new_position

    def check_x_borders(self, hyp_vel, hyp_pos, radius):
        """
        check for collisions with right or left border

        Parameters
        ----------
        hyp_vel : (position_dimensions,) array
        hyp_pos : (position_dimensions,) array
        radius : float

        Returns
        -------
        new_velocity : (position_dimensions,) array
        new_positions : (position_dimensions,) array

        """
        # new velocities and position without border
        new_velocity = np.copy(hyp_vel)
        new_position = np.copy(hyp_pos)

        # -----
        # differences between left and right border
        # -----
        left_diff = hyp_pos[0] - (self.left_border + radius)
        right_diff = hyp_pos[0] - (self.right_border - radius)

        # check borders
        # too far left
        if left_diff <= 0:
            new_velocity[0] = 0.0
            new_velocity[1] = hyp_vel[1] * self.side_friction
            new_position[0] = self.left_border + radius

        # too far right:
        elif right_diff >= 0:
            new_velocity[0] = 0.0
            new_velocity[1] = hyp_vel[1] * self.side_friction
            new_position[0] = self.right_border - radius

        return new_velocity, new_position

    def check_agents(self, hyp_vel, hyp_pos, radius, current_agent_name):
        """
        check for collisions with other agents

        Parameters
        ----------
        hyp_vel : (position_dimensions,) array
        hyp_pos : (position_dimensions,) array
        radius : float
        current_agent_name : str

        Returns
        -------
        new_velocity : (position_dimensions,) array
        new_positions : (position_dimensions,) array

        """
        new_velocity = np.copy(hyp_vel)
        new_position = np.copy(hyp_pos)

        for other_agent in self.env_objects:
            if other_agent['name'] == current_agent_name:
                continue

            # Check if intersection exists
            other_agent_current_position = other_agent['position']

            if np.array_equiv(other_agent_current_position, np.zeros(2)):
                print("wrong")

            distance = np.linalg.norm(other_agent_current_position - hyp_pos)
            if distance > (other_agent['radius'] + radius):
                # No intersection
                continue

            # Get type of intersection:
            # self is at
            # - top right
            # - top left
            # - bottom right
            # - bottom left
            # of other agent
            if hyp_pos[1] <= other_agent_current_position[1]:
                pos = 'bottom'
            else:
                pos = 'top'

            if hyp_pos[0] <= other_agent_current_position[0]:
                pos = pos + ' left'
            else:
                pos = pos + ' right'

            x_diff = abs(other_agent_current_position[0] - hyp_pos[0])
            y_diff = abs(other_agent_current_position[1] - hyp_pos[1])

            z = y_diff / (x_diff + .1e-30)

            a = radius / np.sqrt(z ** 2 + 1)
            b = (z * radius) / np.sqrt(z ** 2 + 1)

            a_other_agent = other_agent['radius'] / np.sqrt(z ** 2 + 1)
            b_other_agent = (z * other_agent['radius']) / np.sqrt(z ** 2 + 1)

            delta_x = a - x_diff + a_other_agent
            delta_y = b - y_diff + b_other_agent

            # Now change position depending on direction
            if 'bottom' in pos:
                new_position[1] -= delta_y
            else:
                new_position[1] += delta_y

            if 'left' in pos:
                new_position[0] -= delta_x
            else:
                new_position[0] += delta_x

            new_velocity[0] = hyp_vel[0] * self.agent_friction
            new_velocity[1] = hyp_vel[1] * self.agent_friction

        return new_velocity, new_position

    def update_targets(self):
        """update targets for all agents in the environment"""
        for env_object in self.env_objects:
            if env_object['type'] == 'agent':
                env_object['target'] = next(env_object['target iterator'])

    def update_sensor_readings(self):
        """update sensor readings for all agents in the environment"""
        for env_object in self.env_objects:
            if env_object['type'] == 'agent':
                self.update_sensor_readings_for_one_agent(env_object)

    def update_sensor_readings_for_one_agent(self, agent_object):
        """
        calculate the sensor readings for the current agent

        in the current environment state

        Parameters
        ----------
        agent_object : dict
            name : str
            type : str
            position': (position_dimensions,) array
            velocity': (position_dimensions,) array
            mass : float
            radius : float
            color : str
            thruster activity : (motor_commands_dimensions,) array
            sensor readings : (sensor_readings_dimensions,) array
            target_type : str
            predictions : (prediction_horizon, position_dimensions) array
            target : (position_dimensions,) array

        Returns
        -------

        """
        current_agent_sensor_readings = \
            util.convert_ray_distances_to_sensor_readings(
                self.calculate_ray_distances(agent_object['position'],
                                             agent_object['radius']))

        current_agent_sensor_readings = \
            self.calculate_distances_to_other_agents(
                agent_object['name'],
                agent_object['position'],
                agent_object['radius'],
                current_agent_sensor_readings)

        if self.point_spread_flag:
            self.apply_point_spread(current_agent_sensor_readings)

        if self.normalize_sensor_readings:
            current_agent_sensor_readings /= \
                np.sum(current_agent_sensor_readings) + 10**(-8)

        agent_object['sensor readings'] = current_agent_sensor_readings

    def calculate_ray_distances(self, current_position, current_radius):
        """
        calculate distances in all directions to all borders

        Parameters
        ----------
        current_position : (position_dimensions,) array
        current_radius : float

        Returns
        -------
        ray_distances : (sensor_readings_dimensions,) array

        """
        # Store the values at the particular rays in array
        ray_distances = np.zeros(self.sensor_readings_dimensions)

        for ray_index in range(self.sensor_readings_dimensions):
            # Calculate distances to all borders
            distances = \
                util.calculate_all_border_distances(
                    current_position,
                    current_radius,
                    self.sensor_directions[ray_index],
                    self.max_distance,
                    self.border_proximity_weight,
                    self.borders
                )
            ray_distances[ray_index] = np.max(distances)
        return ray_distances

    def calculate_distances_to_other_agents(self,
                                            current_name,
                                            current_position,
                                            current_radius,
                                            current_sensor_readings):
        """
        calculate the closest distance to all other agents

        Parameters
        ----------
        current_name : str
        current_position : (position_dimensions,) array
        current_radius : float
        current_sensor_readings : (sensor_readings_dimensions,) array

        Returns
        -------
        current_sensor_readings : (sensor_readings_dimensions,) array

        """
        for other_env_objects in self.env_objects:
            if other_env_objects['type'] == 'agent':
                if other_env_objects['name'] != current_name:
                    other_agent_position = other_env_objects['position']
                    other_agent_radius = other_env_objects['radius']
                    active_sensor_index, distance = \
                        util.calculate_closest_agent_distance(
                            current_position,
                            other_agent_position,
                            current_radius,
                            other_agent_radius,
                            self.sensor_readings_dimensions,
                            self.max_distance)
                    if active_sensor_index is not None:
                        current_sensor_readings[active_sensor_index] = \
                            np.min([current_sensor_readings[
                                       active_sensor_index],
                                   distance])
        return current_sensor_readings

    def apply_point_spread(self, current_agent_sensor_readings):
        """
        apply point spreading onto the current sensor readings

        Parameters
        ----------
        current_agent_sensor_readings : (sensor_readings_dimensions,) array

        Returns
        -------
        point_spread_signal : (sensor_readings_dimensions,) array
            clipped and rounded point spread sensor readings

        """
        point_spread_signal = np.zeros(self.sensor_readings_dimensions)

        for active_sensor_index in range(self.sensor_readings_dimensions):
            point_spread_signal += \
                util.point_spread(
                    active_sensor_index,
                    current_agent_sensor_readings[active_sensor_index],
                    self.point_spread_type,
                    self.sensor_readings_dimensions,
                    self.point_spread_size,
                    self.point_spread_sigma)

        return np.round_(np.clip(point_spread_signal,
                                 self.point_spread_clip_min,
                                 self.point_spread_clip_max),
                         decimals=self.point_spread_round_decimals)
