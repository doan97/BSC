import enum
import math
import random

import numpy as np

import WORKING_MARCO.global_config as c


class RB3Mode(enum.Enum):
    Rocket = 0
    Glider = 1
    Stepper = 2
    Stepper2Dir = 4


MIN_MOTOR_VALUE = 0
MAX_MOTOR_VALUE = 1

NUMBER_OF_PROBLEMS = 1


class Simulator(object):
    def __init__(self, mode=0, max_thrust=1.0,
                 left_border=-1.5, right_border=1.5,
                 top_border=2.0, bot_border=0.0, delta_time=1. / 30.,
                 stopwatch=None):

        self.max_thrust = max_thrust
        self.left_border = left_border
        self.right_border = right_border
        self.top_border = top_border
        self.bot_border = bot_border

        self.borders = np.array(
            [
                [[left_border, bot_border], [left_border, top_border]],
                [[left_border, top_border], [right_border, top_border]],
                [[right_border, top_border], [right_border, bot_border]],
                [[right_border, bot_border], [left_border, bot_border]]
            ]
        )

        self.delta_time = delta_time

        self.sensor_dim = c.INPUT_POSITION_DIM
        self.motor_dim = c.INPUT_MOTOR_DIM

        self.g = 0.0

        self.floor_friction = 0.0
        self.ceiling_friction = 0.0
        self.side_friction = 0.0
        self.agent_friction = 0.0

        tmp1 = np.array([1.0, 1.0])
        tmp2 = np.array([-1.0, 1.0])
        tmp3 = np.array([1.0, -1.0])
        tmp4 = np.array([-1.0, -1.0])
        tmp1_norm = tmp1 / np.linalg.norm(tmp1)
        tmp2_norm = tmp2 / np.linalg.norm(tmp2)
        tmp3_norm = tmp3 / np.linalg.norm(tmp3)
        tmp4_norm = tmp4 / np.linalg.norm(tmp4)
        self.thrustDirections = np.array([tmp1_norm, tmp2_norm, tmp3_norm, tmp4_norm])

        self.stopwatch = stopwatch
        self.sensor_dirs = calc_sensor_dirs()
        self.tmp_counter = 0

        # set mode at last
        self.mode = mode
        self._set_mode(mode)

    def switch_mode(self, do_random=True, mode=None):

        if mode is not None:
            next_problem_index = mode
        elif do_random:
            next_problem_index = int((random.random() * NUMBER_OF_PROBLEMS) // 1)
        else:
            next_problem_index = (self.mode + 1) % NUMBER_OF_PROBLEMS

        self._set_mode(next_problem_index)

    def _set_mode(self, mode):
        self.mode = mode
        if mode == RB3Mode.Rocket.value:
            self.g = 9.81

            self.floor_friction = .9
            self.ceiling_friction = .9
            self.side_friction = .9
            self.agent_friction = .9

            self.numThrusts = 2

        elif mode == RB3Mode.Glider.value:
            self.g = 0.0

            self.floor_friction = .9
            self.ceiling_friction = .9
            self.side_friction = .9
            self.agent_friction = .4

            self.numThrusts = 4

        elif mode == RB3Mode.Stepper.value:
            self.g = 0.0

            self.floor_friction = 1
            self.ceiling_friction = 1
            self.side_friction = 1
            self.agent_friction = 1

            self.numThrusts = 4

        else:
            raise SystemExit("SIMULATOR MODE UNKNOWN")

    def update(self, previous_velocity, previous_position, motor_commands, current_agent, time_step, obstacles, borders=True):
        """
        Takes a simulator state (previous position and velocity) and motor commands

        Returns the new position and position delta after running these commands

        Parameters
        ----------
        previous_velocity : ...
        previous_position : ...
        motor_commands : ...
        current_agent : ...
        time_step : ...

        Returns
        -------
        ...
        """
        mass = current_agent.mass
        gravity = np.array([0.0, -1.0]) * mass * self.g

        if self.stopwatch is not None:
            self.stopwatch.start('sim')

        if not isinstance(motor_commands, np.ndarray) or not (
                len(motor_commands.shape) == 1 and motor_commands.shape[0] == 4):
            raise ValueError(
                "MotorCommands must be a 1-dimensional numpy-Array with 4 elements: shape = (4,) but shape is " + str(
                    motor_commands.shape))

        thrusts = self.get_thrusts_from_motor_commands(motor_commands)

        # calculate the force that is applied in x an y direction
        # orientation of thrusters is important
        thrust_forces = np.matmul((thrusts * self.max_thrust)[np.newaxis, :], self.thrustDirections)

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
        hypvel = previous_velocity + velocity_change

        # consequent position without borders
        # delta_s = v * delta_t
        # s = s_0 + delta_s
        position_change = hypvel * self.delta_time
        hyppos = previous_position + position_change

        # Calculate velocity and position after consideration of borders and other agents
        if borders:
            new_velocity, new_position = self.check_collisions(hypvel, hyppos, current_agent.radius,
                                                            obstacles, time_step)
        else:
            new_velocity, new_position = hypvel, hyppos

        if self.mode in (RB3Mode.Stepper2Dir.value, RB3Mode.Stepper.value):
            new_velocity = np.zeros(self.sensor_dim)

        # ================================
        # Now calculate the real Position
        # ================================
        # In delta-Mode, the Simulator and the NN calculate the difference in position
        # between the previous and the current timestep.
        # The Input for the next NN-iteration however must be the real new position and not the delta.
        # That's why sensorOutput and nextSensorInput are separated

        position_delta = new_position - previous_position
        absolute_position = new_position

        if self.stopwatch is not None:
            self.stopwatch.stop('sim')

        # if not (all(np.equal(np.around(position_delta, 10), np.around(position_change, 10)))):
        #     print("Position_delta != position_change" + str(self.tmp_counter))
        #     self.tmp_counter += 1

        return position_delta, absolute_position, new_velocity, acceleration

    def check_collisions(self, hyp_vel, hyp_pos, radius, other_agents, time_step):
        hyp_vel, hyp_pos = self.check_y_border_collisions(hyp_vel, hyp_pos, radius,
                                                          self.top_border, self.bot_border,
                                                          self.ceiling_friction, self.floor_friction)
        hyp_vel, hyp_pos = self.check_x_border_collisions(hyp_vel, hyp_pos, radius,
                                                          self.left_border, self.right_border,
                                                          self.side_friction, self.side_friction)
        hyp_vel, hyp_pos = self.check_agent_collisions(hyp_vel, hyp_pos, radius, other_agents, time_step,
                                                       self.agent_friction)

        return hyp_vel, hyp_pos

    @staticmethod
    def check_y_border_collisions(hyp_vel, hyp_pos, radius, top_border, bot_border, top_friction, bot_friction):
        new_velocity = np.copy(hyp_vel)
        new_position = np.copy(hyp_pos)

        # new velocities and position without border
        # -----
        # differences between bottom and top border
        # -----
        bottom_diff = hyp_pos[1] - (bot_border + radius)
        top_diff = hyp_pos[1] - (top_border - radius)

        # check borders
        # too far down
        if bottom_diff <= 0:
            new_velocity[0] = hyp_vel[0] * bot_friction
            new_velocity[1] = 0.0
            new_position[1] = bot_border + radius

        # too far up
        elif top_diff >= 0:
            new_velocity[0] = hyp_vel[0] * top_friction
            new_velocity[1] = 0.0
            new_position[1] = top_border - radius

        return new_velocity, new_position

    @staticmethod
    def check_x_border_collisions(hyp_vel, hyp_pos, radius, left_border, right_border, left_friction, right_friction):
        new_velocity = np.copy(hyp_vel)
        new_position = np.copy(hyp_pos)

        # new velocities and position without border
        # -----
        # differences between left and right border
        # -----
        left_diff = hyp_pos[0] - (left_border + radius)
        right_diff = hyp_pos[0] - (right_border - radius)

        # check borders
        # too far left
        if left_diff <= 0:
            new_velocity[0] = 0.0
            new_velocity[1] = hyp_vel[1] * left_friction
            new_position[0] = left_border + radius

        # too far right:
        elif right_diff >= 0:
            new_velocity[0] = 0.0
            new_velocity[1] = hyp_vel[1] * right_friction
            new_position[0] = right_border - radius

        return new_velocity, new_position

    @staticmethod
    def check_agent_collisions(hyp_vel, hyp_pos, radius, other_agents, time_step, agent_friction):
        new_velocity = np.copy(hyp_vel)
        new_position = np.copy(hyp_pos)

        # Shuffle agent list
        random.shuffle(other_agents)
        for other_agent in other_agents:
            # Check if intersection exists
            #current_position_other_agent = other_agent.data.positions.get(time_step)
            current_position_other_agent = other_agent['position']

            distance = np.linalg.norm(current_position_other_agent - hyp_pos)
            if distance > (other_agent['radius'] + radius):
                # No intersection
                continue

            difference = np.abs(current_position_other_agent - hyp_pos)

            z = difference[1] / (difference[0] + .1e-30)

            a = radius / math.sqrt(z ** 2 + 1)
            b = (z * radius) / math.sqrt(z ** 2 + 1)

            other_agent_a = other_agent['radius'] / math.sqrt(z ** 2 + 1)
            other_agent_b = (z * other_agent['radius']) / math.sqrt(z ** 2 + 1)

            delta_x = a - difference[0] + other_agent_a
            delta_y = b - difference[1] + other_agent_b

            # change position depending on direction
            if hyp_pos[1] <= current_position_other_agent[1]:
                new_position[1] -= delta_y
            else:
                new_position[1] += delta_y

            if hyp_pos[0] <= current_position_other_agent[0]:
                new_position[0] -= delta_x
            else:
                new_position[0] += delta_x

            new_velocity = hyp_vel * agent_friction

        return new_velocity, new_position

    @staticmethod
    def calc_sensor_data(from_step, to_step, agent, sensor_dirs, borders, position, other_agents):
        time_steps = to_step - from_step
        sensor_data = np.zeros([time_steps, c.INPUT_SENSOR_DIM])
        for t in range(time_steps):
            my_pos = position
            my_radius = agent.radius
            ray_proximities = calculate_ray_proximities(my_pos, my_radius, sensor_dirs, borders)
            sensor_data[t] = convert_ray_proximities_to_sensor_data(ray_proximities)
            sensor_data[t] = calculate_distances_to_other_agents(other_agents,
                                                                 my_pos, my_radius, sensor_data[t], from_step + t)
            # Apply point spread function
            if c.POINT_SPREAD:
                sensor_data[t] = calculate_point_spread_signal(sensor_data[t])
        return sensor_data

    def calc_sensor_data2(from_step, to_step, position, sensor_dirs, borders):
        time_steps = to_step - from_step

        sensor_data = np.zeros([time_steps, c.INPUT_SENSOR_DIM])

        for t in range(time_steps):

            my_pos = position
            my_radius = 0.06

            ray_proximities = calculate_ray_proximities(my_pos, my_radius, sensor_dirs, borders)

            sensor_data[t] = convert_ray_proximities_to_sensor_data(ray_proximities)

            sensor_data[t] = calculate_distances_to_other_agents(agent.other_agents,
                                                                 my_pos, my_radius, sensor_data[t], from_step + t)

            # Apply point spread function
            if c.POINT_SPREAD:
                sensor_data[t] = calculate_point_spread_signal(sensor_data[t])

        return sensor_data

    def simulate_multiple(self, previous_position, previous_velocity, all_motor_commands, current_agent, from_step):
        """
        Simulates the give motor commands for multiple time steps (as many as motor commands exist)
        and returns the resulting positions.
        Starts at a given state.

        Parameters
        ----------
        previous_position : ...
        previous_velocity : ...
        all_motor_commands : ...
        current_agent : ...
        from_step : ...

        Returns
        -------
        ...
        """
        assert (all_motor_commands.shape[1] == self.motor_dim), "Simulated motor commands not in right shape"

        all_sensor_outputs = []
        all_next_sensor_inputs = []
        all_velocities = []

        for motor_commands in all_motor_commands:
            # Perform update
            sensor_output, next_sensor_input, new_velocity, _ = self.update(previous_velocity, previous_position,
                                                                            motor_commands, current_agent, from_step)
            all_sensor_outputs.append(sensor_output)
            all_next_sensor_inputs.append(next_sensor_input)
            all_velocities.append(new_velocity)

        all_sensor_outputs = np.asarray(all_sensor_outputs)
        all_next_sensor_inputs = np.asarray(all_next_sensor_inputs)
        all_velocities = np.asarray(all_velocities)

        # Return
        return all_sensor_outputs, all_next_sensor_inputs, all_velocities

    def get_random_motor_commands(self, last_commands):
        motor_commands = np.zeros(self.motor_dim)

        r = random.random()

        if r < 0.7:  # 70%: new commands

            r = random.random()

            if self.mode == 0:
                # 1 of 2 active .2
                # both active .4
                # both equal .3
                # all out .1

                if r < 0.1:  # 10%: do nothing
                    motor_commands = np.zeros(self.motor_dim)

                elif r < 0.3:  # 20%: 1 active
                    active_thrust = int(random.random() * self.numThrusts)
                    motor_commands[active_thrust] = random.random()

                elif r < 0.7:  # 20%: both active, but random
                    motor_commands = np.random.random(size=self.motor_dim)

                else:  # 30%: both active, but equal
                    motor_commands[0] = np.random.random()
                    motor_commands[1] = motor_commands[0]

            else:
                # 0 of 4 active .2
                # 1 of 4 active .2
                # 2 of 4 active .2
                # 3 of 4 active .2
                # 4 of 4 active .2

                if r < 0.2:  # 20%: do nothing
                    motor_commands = np.zeros(self.motor_dim)

                elif r < 0.4:  # 40%: 1 active
                    active_thrust = int(random.random() * self.numThrusts)
                    motor_commands[active_thrust] = random.random()

                elif r < 0.6:  # 20%: 2 active
                    active_thrust_1 = int(random.random() * self.numThrusts)

                    active_thrust_2 = active_thrust_1
                    while active_thrust_1 == active_thrust_2:
                        active_thrust_2 = int(random.random() * self.numThrusts)

                    motor_commands[active_thrust_1] = random.random()
                    motor_commands[active_thrust_2] = random.random()

                elif r < 0.8:  # 20%: 3 active
                    active_thrust_1 = int(random.random() * self.numThrusts)

                    active_thrust_2 = active_thrust_1
                    while active_thrust_1 == active_thrust_2:
                        active_thrust_2 = int(random.random() * self.numThrusts)

                    active_thrust_3 = active_thrust_2
                    while active_thrust_2 == active_thrust_3:
                        active_thrust_3 = int(random.random() * self.numThrusts)

                    motor_commands[active_thrust_1] = random.random()
                    motor_commands[active_thrust_2] = random.random()
                    motor_commands[active_thrust_3] = random.random()

                else:  # 20%: all active
                    motor_commands = np.random.random(size=self.motor_dim)

        else:
            motor_commands = last_commands

        return motor_commands

    def get_thrusts_from_motor_commands(self, motor_commands):
        motor_commands = np.clip(motor_commands, MIN_MOTOR_VALUE, MAX_MOTOR_VALUE)

        thrusts = np.zeros(self.motor_dim)
        thrusts[0:2] = motor_commands[0:2]

        if self.mode in (RB3Mode.Glider.value, RB3Mode.Stepper.value):
            thrusts[2:4] = motor_commands[2:4]
        # else:
        # Nur fuer Stepper2Dir
        # changeInOrientation = (motorCommands[3] - motorCommands[2]) * np.pi

        return thrusts


# ----------------------------
# Begin of Helper Functions
# ----------------------------


def calculate_ray_proximities(my_pos, my_radius, sensor_dirs, borders):
    # Store the values at the particular rays in array
    ray_proximities = np.zeros([c.INPUT_SENSOR_DIM])

    for ray_index in range(c.INPUT_SENSOR_DIM):
        ray_proximities[ray_index] = \
            calculate_border_ray_proximities(my_pos, my_radius, sensor_dirs[ray_index], borders)
    return ray_proximities


def calculate_border_ray_proximities(my_pos, my_radius, sensor_direction, borders):
    # Calculate proximity to each border
    proximities = [0.0]
    for border in borders:
        proximities.append(calc_border_intersection_proximity(my_pos, my_radius, border, sensor_direction))
    return max(proximities)


def calc_border_intersection_proximity(my_pos, my_radius, border, sensor_vector):
    """
    calculate the weighted linear proximity value starting from position to given border in vector direction

    returns none, if direction and border have no intersection in interval of interest

    Parameters
    ----------
    my_pos : ...
    my_radius : ...
    border : ...
    sensor_vector : ...

    Returns
    -------
    ...

    """
    intersection, valid = calculate_line_line_intersection(my_pos, my_pos + sensor_vector, border[0], border[1])

    if valid:
        distance = np.linalg.norm(intersection - my_pos)
        real_distance = distance - my_radius
        proximity = get_proximity(real_distance, my_radius)
        return max(proximity * c.BORDER_PROXIMITY_WEIGHT, 0.0)
    else:
        return -1.0


def calculate_line_line_intersection(vector_one_start, vector_one_end, vector_two_start, vector_two_end, check='two'):
    """

    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    https://mathworld.wolfram.com/Line-LineIntersection.html

    Parameters
    ----------
    vector_one_start : (2,) array
    vector_one_end : (2,) array
    vector_two_start : (2,) array
    vector_two_end : (2,) array
    check : str


    Returns
    -------
    intersection : (2,) array
    valid : bool

    """
    x1, y1 = vector_one_start
    x2, y2 = vector_one_end
    x3, y3 = vector_two_start
    x4, y4 = vector_two_end

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if np.allclose(0, denominator):
        return None, False

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = - ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

    t = np.round(t, 6)
    u = np.round(u, 6)

    p = vector_one_start + t * (vector_one_end - vector_one_start)

    # np.testing.assert_almost_equal(p, vector_two_start + u * (vector_two_end - vector_two_start))

    if check == 'one':
        if 0 <= t <= 1 and u >= 0:
            return p, True

        else:
            return p, False

    elif check == 'two':
        if 0 <= u <= 1 and t >= 0:
            return p, True

        else:
            return p, False

    else:
        return p, None


def convert_ray_proximities_to_sensor_data(ray_proximities):
    # Convert the ray proximities to sensor data
    # A sensor's proximity is the max of its two surrounding rays
    return np.maximum.reduce([ray_proximities, np.roll(ray_proximities, -1)])


def calculate_distances_to_other_agents(other_agents, my_pos, my_radius, sensor_data, time_step):
    # calculate the closest proximity to all other agents from given position
    for a in other_agents:
        other_pos = a['position']
        other_radius = a['radius']
        sensor_index, proximity = calculate_closest_agent_proximity(my_pos, other_pos, my_radius, other_radius)
        if sensor_index is not None and sensor_data[sensor_index] < proximity:
            sensor_data[sensor_index] = proximity
    return sensor_data


def calculate_closest_agent_proximity(my_pos, other_pos, my_radius, other_radius):
    """
    this function calculates and returns the proximity to a particular agent

    and the index of the sensor that can sense the closest part of the agent.

    0 = not sensing anything; high value = high proximity

    Parameters
    ----------
    my_pos : ...
    other_pos : ...
    my_radius : ...
    other_radius : ...

    Returns
    -------
    ...
    """
    distance = np.linalg.norm(other_pos - my_pos)
    real_distance = max(distance - my_radius - other_radius, 0)

    proximity = get_proximity(real_distance, my_radius)

    if proximity == 0:
        return None, 0.0

    x_diff = other_pos[0] - my_pos[0]
    y_diff = other_pos[1] - my_pos[1]

    # measure rotation in multiple of pi
    angle_rad = np.arctan2(y_diff, x_diff) / np.pi

    # add semi circle when negative rotation direction is calculated
    if angle_rad < 0:
        angle_rad += 2

    # Determine the id of the active sensor, clockwise
    active_sensor = int((angle_rad * c.INPUT_SENSOR_DIM) // 2)

    # prevent invalid index if the angle is exactly 360 degrees
    active_sensor = min(active_sensor, c.INPUT_SENSOR_DIM - 1)

    return active_sensor, proximity


def get_proximity(distance, radius, mode='linear'):
    max_distance = c.MAX_DISTANCE * radius
    if mode == 'linear':
        return _get_proximity_linear(distance, max_distance)
    elif mode == 'tanh':
        return _get_proximity_tanh(distance, max_distance)
    else:
        raise ValueError


def _get_proximity_tanh(distance, max_distance):
    # Ignore sensor data when distance is too high
    return max(np.tanh(1 / (distance + 1e-30) - 1 / max_distance) + 1e-30, 0.0)


def _get_proximity_linear(distance, max_distance):
    # Ignore sensor data when distance is too high
    return max(- 1.0 * (distance / max_distance) + 1.0, 0.0)


def calculate_point_spread_signal(sensor_data):
    point_spread_signal = np.zeros(sensor_data.shape)
    for active_sensor_index, sensor_reading in enumerate(sensor_data):
        point_spread_signal += point_spread(active_sensor_index, sensor_reading, c.POINT_SPREAD_TYPE)

    return np.round_(
        np.clip(
            point_spread_signal + sensor_data, 0, 1
        ), decimals=8
    )


def point_spread(active_sensor_index, sensor_proximity_reading, function_type):
    """
    ...

    Parameters
    ----------
    active_sensor_index : ...
    sensor_proximity_reading : ...
    function_type : ...

    Returns
    -------
    ...
    """
    steps_ = np.arange(-(c.INPUT_SENSOR_DIM - 1) // 2, c.INPUT_SENSOR_DIM // 2)
    shift = -(c.INPUT_SENSOR_DIM - 1) // 2 + active_sensor_index
    in_ = np.roll(steps_, shift)

    if function_type == 'linear' or function_type == 'linear_normalized':
        out_ = (-1 * c.SPREADSIZE) * np.abs(in_) + 1.0
    elif function_type == 'gauss':
        out_ = np.exp(-(np.abs(in_) ** 2) / (2 * c.SIGMA ** 2))
    else:
        raise RuntimeError

    point_spread_ = out_ * sensor_proximity_reading

    if function_type is 'linear_normalized':
        point_spread_ /= np.sum(point_spread_) + 10 ** (-8)

    point_spread_[active_sensor_index] = 0

    return point_spread_


def calc_sensor_dirs():
    dirs = []
    for i in range(c.INPUT_SENSOR_DIM):
        sensor_range_rad = (2 * np.pi) / c.INPUT_SENSOR_DIM
        angle_rad = (i * sensor_range_rad + (i + 1) * sensor_range_rad) / 2.
        dirs.append(np.array([np.cos(angle_rad), np.sin(angle_rad)]))

    dirs = np.asarray(dirs)
    return dirs
