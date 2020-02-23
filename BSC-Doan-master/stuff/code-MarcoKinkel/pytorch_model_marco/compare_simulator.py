import numpy as np
from enum import Enum
import math
import time
import random

import global_config as c


class RB3Mode(Enum):
    Rocket = 0
    Glider = 1
    Stepper = 2
    Stepper2Dir = 4


MIN_MOTOR_VALUE = 0
MAX_MOTOR_VALUE = 1

NUMBER_OF_PROBLEMS = 1


class Simulator(object):
    def __init__(self, mode=0, maxthrust=1.0, leftborder=-1.5, rightborder=1.5, topborder=2.0, delta_time=1. / 30.,
                 stopwatch=None):
        self.use_borders = False
        self.maxthrust = maxthrust
        self.leftborder = leftborder
        self.rightborder = rightborder
        self.topborder = topborder
        self.delta_time = delta_time

        self.sensor_dim = c.INPUT_POSITION_DIM
        self.motor_dim = c.INPUT_MOTOR_DIM

        self.g = 0.0

        self.floor_friction = 0.0
        self.ceilingFriction = 0.0
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

        # Ganz am Ende, damit Variablen nicht wieder ueberschrieben werden:
        self.mode = mode
        self._set_mode(mode)

        self.sensor_dirs = calc_sensor_dirs()

        self.tmp_counter = 0

    # Ende __init__

    def __reset(self, mode=None):
        self.previous_position = self.current_position = np.array([self.initPos[0], self.initPos[1]])
        self.velocity = np.zeros(self.sensor_dim)

        if mode is not None:
            self.switch_mode(mode=mode)

    # '''
    # Returns a Simulator_State object. Needs the initial position.
    # '''
    # def get_initial_state(self, init_pos):
    #     initial_velocity = 0.0
    #     initial_position = init_pos
    #     previous_position = init_pos
    #     return Simulator_State(initial_position, previous_position, initial_velocity)

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
            self.ceilingFriction = .9
            self.side_friction = .9
            self.agent_friction = .9

            self.numThrusts = 2


        elif mode == RB3Mode.Glider.value:
            self.g = 0.0

            self.floor_friction = .9
            self.ceilingFriction = .9
            self.side_friction = .9
            self.agent_friction = .4

            self.numThrusts = 4


        elif mode == RB3Mode.Stepper.value:
            self.g = 0.0

            self.floor_friction = 1
            self.ceilingFriction = 1
            self.side_friction = 1
            self.agent_friction = 1

            self.numThrusts = 4

        else:
            raise SystemExit("SIMULATOR MODE UNKNOWN")

    '''
    Takes a simulator state (previous position and velocity) and motor commands
    Returns the new position and position delta after running these commands
    '''

    def update(self, previous_velocity, previous_position, motorCommands, current_agent, time_step):

        mass = current_agent.mass
        gravity = np.array([0.0, -1.0]) * mass * self.g

        if self.stopwatch is not None:
            self.stopwatch.start('sim')

        if not isinstance(motorCommands, np.ndarray) or not (
                len(motorCommands.shape) == 1 and motorCommands.shape[0] == 4):
            raise ValueError(
                "MotorCommands must be a 1-dimensional numpy-Array with 4 elements: shape = (4,) but shape is " + str(
                    motorCommands.shape))

        thrusts = self.get_thrusts_from_motor_commands(motorCommands)

        # Berechne die Kraft, die in x und y-Richtung ausgeuebt wird (Orientierung der Duesen ist wichtig)
        thrust_forces = np.matmul((thrusts * self.maxthrust)[np.newaxis, :], self.thrustDirections)

        # Verrechne die von den Duesen ausgehende Kraft mit bereits bestehenden Kraeften wie Gravitation
        forcesum = thrust_forces[0] + gravity

        # a = F/m
        acceleration = forcesum / mass

        # consequent velocity without borders.
        # v = delta_s / delta_t
        # or
        # delta_v = a * delta_t
        # v = v_0 + delta_v
        velocity_change = acceleration * self.delta_time
        hypvel = previous_velocity + velocity_change #REAL VELOCITY

        # consequent position without borders
        # delta_s = v * delta_t
        # s = s_0 + delta_s
        position_change = hypvel * self.delta_time
        hyppos = previous_position + position_change #REAL POSITION

        if self.use_borders:
            # Calculate velocity and position after consideration of borders and other agents
            new_velocity, new_position = self.check_borders(hypvel, hyppos, current_agent.radius,
                                                            current_agent.other_agents, time_step)
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

    def check_borders(self, hypvel, hyppos, radius, other_agents, time_step):

        # functions = []
        # functions.append(self.check_y_borders)
        # functions.append(self.check_x_borders)
        # functions.append(self.check_agents)

        # # Do this in random order
        # random.shuffle(functions)
        # for fn in functions:
        #     hypvel, hyppos = fn(hypvel, hyppos, radius, other_agents)

        hypvel, hyppos = self.check_y_borders(hypvel, hyppos, radius)
        hypvel, hyppos = self.check_x_borders(hypvel, hyppos, radius)
        hypvel, hyppos = self.check_agents(hypvel, hyppos, radius, other_agents, time_step)

        return hypvel, hyppos

    def check_y_borders(self, hypvel, hyppos, radius):
        new_velocity = np.copy(hypvel)
        new_position = np.copy(hyppos)

        # new velocities and position without border
        # -----
        # differences between bottom and top border
        # -----
        bottom_diff = hyppos[1] - radius  # bottom border = 0
        top_diff = hyppos[1] - (self.topborder - radius)

        # check borders
        if bottom_diff <= 0:  # down at floor border
            new_velocity[0] = hypvel[0] * self.floor_friction
            new_velocity[1] = 0.0
            new_position[1] = radius

        elif top_diff >= 0:  # up at top ceiling border
            new_velocity[0] = hypvel[0] * self.ceilingFriction
            new_velocity[1] = 0.0
            new_position[1] = self.topborder - radius

        return new_velocity, new_position

    def check_x_borders(self, hypvel, hyppos, radius):
        new_velocity = np.copy(hypvel)
        new_position = np.copy(hyppos)

        # -----
        # differences between left and right border
        # -----
        left_diff = hyppos[0] - (self.leftborder + radius)
        right_diff = hyppos[0] - (self.rightborder - radius)

        # check borders
        # too far left
        if left_diff <= 0:
            new_velocity[0] = 0.0
            new_velocity[1] = hypvel[1] * self.side_friction
            new_position[0] = self.leftborder + radius

        # too far right:
        elif right_diff >= 0:
            new_velocity[0] = 0.0
            new_velocity[1] = hypvel[1] * self.side_friction
            new_position[0] = self.rightborder - radius

        return new_velocity, new_position

    def check_agents(self, hypvel, hyppos, radius, other_agents, time_step):
        new_velocity = np.copy(hypvel)
        new_position = np.copy(hyppos)

        # Shuffle agent list
        random.shuffle(other_agents)

        for A in other_agents:
            # Check if intersection exists
            A_current_position = A.data.positions.get(time_step)

            if np.array_equiv(A_current_position, np.zeros(2)):
                # print("FALSCH")
                pass

            distance = np.linalg.norm(A_current_position - hyppos)
            if distance > (A.radius + radius):
                # No intersection
                continue

            # Get type of intersection:
            # self is at
            # - top right
            # - top left
            # - bottom right
            # - bottom left
            # of A
            if hyppos[1] <= A_current_position[1]:
                pos = 'bottom'
            else:
                pos = 'top'

            if hyppos[0] <= A_current_position[0]:
                pos = pos + ' left'
            else:
                pos = pos + ' right'

            xdiff = abs(A_current_position[0] - hyppos[0])
            ydiff = abs(A_current_position[1] - hyppos[1])

            z = ydiff / (xdiff + .1e-30)

            a = radius / math.sqrt(z ** 2 + 1)
            b = (z * radius) / math.sqrt(z ** 2 + 1)

            a_A = A.radius / math.sqrt(z ** 2 + 1)
            b_A = (z * A.radius) / math.sqrt(z ** 2 + 1)

            delta_x = a - xdiff + a_A
            delta_y = b - ydiff + b_A

            # Now change position depending on direction
            if 'bottom' in pos:
                new_position[1] -= delta_y
            else:
                new_position[1] += delta_y

            if 'left' in pos:
                new_position[0] -= delta_x
            else:
                new_position[0] += delta_x

            new_velocity[0] = hypvel[0] * self.agent_friction
            new_velocity[1] = hypvel[1] * self.agent_friction

        return new_velocity, new_position

    def calc_sensor_data(self, from_step, to_step, agent):
        time_steps = to_step - from_step

        sensor_data = np.zeros([time_steps, c.INPUT_SENSOR_DIM])

        for t in range(time_steps):

            my_pos = agent.data.positions.get(from_step + t)
            my_radius = agent.radius

            # Store the values at the particular rays in array
            ray_proximities = np.zeros([c.INPUT_SENSOR_DIM])

            for ray_index in range(c.INPUT_SENSOR_DIM):

                # Calculate intersection of particular ray with all other agents
                # for a in agent.other_agents:
                #     other_pos = a.data.positions.get(from_step + t)
                #     other_radius = a.radius
                #     proximity = calc_agent_intersection_proximity(my_pos, my_radius, other_pos, other_radius, self.sensor_dirs[ray_index])
                #     if (proximity is not None) and (ray_proximities[ray_index] < proximity):
                #         ray_proximities[ray_index] = proximity

                if c.BORDER_PROXIMITY_WEIGHT > 0:
                    # Calculate proximity to each border
                    # Left border
                    border_left = np.array([[-1.5, 0], [-1.5, 2]])
                    proximity = calc_border_intersection_proximity(my_pos, my_radius, border_left,
                                                                   self.sensor_dirs[ray_index])
                    if (proximity is not None) and (ray_proximities[ray_index] < proximity):
                        ray_proximities[ray_index] = proximity

                    # Right border
                    border_right = np.array([[1.5, 0], [1.5, 2]])
                    proximity = calc_border_intersection_proximity(my_pos, my_radius, border_right,
                                                                   self.sensor_dirs[ray_index])
                    if (proximity is not None) and (ray_proximities[ray_index] < proximity):
                        ray_proximities[ray_index] = proximity

                    # Top border
                    border_top = np.array([[-1.5, 2], [1.5, 2]])
                    proximity = calc_border_intersection_proximity(my_pos, my_radius, border_top,
                                                                   self.sensor_dirs[ray_index])
                    if (proximity is not None) and (ray_proximities[ray_index] < proximity):
                        ray_proximities[ray_index] = proximity

                    # Bottom border
                    border_bottom = np.array([[-1.5, 0], [1.5, 0]])
                    proximity = calc_border_intersection_proximity(my_pos, my_radius, border_bottom,
                                                                   self.sensor_dirs[ray_index])
                    if (proximity is not None) and (ray_proximities[ray_index] < proximity):
                        ray_proximities[ray_index] = proximity

            # Convert the ray proximities to the whole sensor
            # A sensor's proximity is the max of its two surrounding rays
            for sensor_index in range(c.INPUT_SENSOR_DIM):
                if sensor_index == c.INPUT_SENSOR_DIM - 1:
                    sensor_data[t, sensor_index] = max([ray_proximities[sensor_index], ray_proximities[0]])
                else:
                    sensor_data[t, sensor_index] = ray_proximities[sensor_index: sensor_index + 2].max()

            # In the end calculate the closest proximity to all agents
            for a in agent.other_agents:
                other_pos = a.data.positions.get(from_step + t)
                other_radius = a.radius
                sensor_index, proximity = calculate_closest_agent_proximity(my_pos, other_pos, my_radius, other_radius)
                if sensor_index is not None and sensor_data[t, sensor_index] < proximity:
                    sensor_data[t, sensor_index] = proximity

            # If the sensor data are not all zero
            if np.any(sensor_data[t]):

                # Apply point spread function
                if c.POINT_SPREAD is True:
                    point_spread_signal = np.zeros([c.INPUT_SENSOR_DIM])

                    for sensor_index in range(c.INPUT_SENSOR_DIM):
                        point_spread_signal_tmp = point_spread(sensor_index, sensor_data[t, sensor_index],
                                                               c.POINT_SPREAD_TYPE)
                        point_spread_signal_tmp[sensor_index] = 0
                        point_spread_signal += point_spread_signal_tmp

                    # Only after all point spread additions is calculated, add it to the sensor data
                    sensor_data[t] += point_spread_signal
                    sensor_data[t] = np.clip(sensor_data[t], 0, 1)
                    sensor_data[t] = np.round(sensor_data[t], decimals=8)

        return sensor_data

    def calc_sensor_data_old(self, from_step, to_step, agent):
        time_steps = to_step - from_step

        sensor_data = np.zeros([time_steps, c.INPUT_SENSOR_DIM])

        for t in range(time_steps):

            # Calculate sensor input for other agents
            for a in agent.other_agents:
                if a.data.positions.curr_idx > (from_step + t) % a.data.positions.length:
                    # Only use the position if the agent's position at this time step has been calculated yet
                    # Otherwise it's the default value, namely [0, 0]

                    # TODO: Do we need to check if my own position has been calculated yet? Depends on when this fn is called
                    my_pos = agent.data.positions.get(from_step + t)
                    other_pos = a.data.positions.get(from_step + t)

                    if (all(np.equal(other_pos, np.zeros(2)))):
                        print("HIER LaeUFT WAS FALSCH")

                    other_radius = a.radius

                    active_sensor, proximity = self.calculate_sensor_input(my_pos, other_pos, agent.radius,
                                                                           other_radius)

                    if active_sensor is None:
                        continue

                    # Apply point spread function to distribute signal over adjacent sensors
                    if c.POINT_SPREAD is True:
                        data = point_spread(active_sensor, proximity, c.POINT_SPREAD_TYPE)

                    for s in range(len(data)):
                        # Check if the sensor already senses a closer target
                        if data[s] > sensor_data[t, s]:
                            # Update the proximity
                            sensor_data[t, s] = np.round(data[s], decimals=5)

            # Calculate sensor input for edges
            if c.PERCEIVE_BORDERS is True:
                my_pos = agent.data.positions.get(from_step + t)
                # edge_points = []

                # sensor_radius = agent.radius * c.MAX_DISTANCE
                # for x in np.arange(-1.5, 1.5, c.BORDER_POINT_DISTANCE):
                #     if my_pos[1] <= sensor_radius:
                #         edge_points.append(np.array([x, 0.]))
                #     elif my_pos[1] >= (2. - sensor_radius):
                #         edge_points.append(np.array([x, 2.]))

                # for y in np.arange(0, 2, c.BORDER_POINT_DISTANCE):
                #     if my_pos[0] <= (-1.5 + sensor_radius):
                #         edge_points.append(np.array([-1.5, y]))
                #     elif my_pos[0] >= (1.5 - sensor_radius):
                #         edge_points.append(np.array([1.5, y]))

                # # print(str(len(edge_points)))

                # for edge_pos in edge_points:
                #     active_sensor, proximity = self.calculate_sensor_input(my_pos, edge_pos, agent.radius, 0.)

                #     if active_sensor is None:
                #         continue

                #     # Apply point spread function to distribute signal over adjacent sensors
                #     data = point_spread(active_sensor, proximity, 'gauss')

                #     for s in range(len(data)):
                #         # Check if the sensor already senses a closer target
                #         if data[s] > sensor_data[t, s]:
                #             # Update the proximity
                #             sensor_data[t, s] = np.round(data[s], decimals=5)

        return sensor_data

        # Fuer jeden Strahl mit Index i:
        #   Berechne Intersection und speichere in strahlen[i]
        #   Falls strahlen[i] == 0 && strahlen[i-1] > 0:
        #       break

        # Fuer jeden Sensor mit Index i:
        #   Berechne Naehe als max(strahlen[i-1,i])

    def __set_savepoint(self):
        self.savepoint_velocity = self.velocity
        self.savepoint_current_position = self.current_position
        self.savepoint_previous_position = self.previous_position

    def __restore_savepoint(self):
        self.velocity = self.savepoint_velocity
        self.current_position = self.savepoint_current_position
        self.savepoint_previous_position = self.savepoint_previous_position

    '''
    Simulates the give motor commands for multiple time steps (as many as motor commands exist)
    and returns the resulting positions.
    Starts at a given state.
    '''

    def simulate_multiple(self, previous_position, previous_velocity, all_motor_commands, current_agent, from_step):

        time_steps = all_motor_commands.shape[0]
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
        return (all_sensor_outputs, all_next_sensor_inputs, all_velocities)

    def get_random_motor_commands(self, last_commands):
        motor_commands = np.zeros(self.motor_dim)

        r = random.random()

        if (r < 0.7):  # 70%: new commands

            r = random.random()

            if self.mode == 0:
                # 1 of 2 active .2
                # both active .4
                # both equal .3
                # all out .1

                if (r < 0.1):  # 10%: do nothing
                    motor_commands = np.zeros(self.motor_dim)

                elif (r < 0.3):  # 20%: 1 active
                    active_thrust = int(random.random() * self.numThrusts)
                    motor_commands[active_thrust] = random.random()

                elif (r < 0.7):  # 20%: both active, but random
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

                if (r < 0.2):  # 20%: do nothing
                    motor_commands = np.zeros(self.motor_dim)

                elif (r < 0.4):  # 40%: 1 active
                    active_thrust = int(random.random() * self.numThrusts)
                    motor_commands[active_thrust] = random.random()

                elif (r < 0.6):  # 20%: 2 active
                    active_thrust_1 = int(random.random() * self.numThrusts)

                    active_thrust_2 = active_thrust_1
                    while (active_thrust_1 == active_thrust_2):
                        active_thrust_2 = int(random.random() * self.numThrusts)

                    motor_commands[active_thrust_1] = random.random()
                    motor_commands[active_thrust_2] = random.random()

                elif (r < 0.8):  # 20%: 3 active
                    active_thrust_1 = int(random.random() * self.numThrusts)

                    active_thrust_2 = active_thrust_1
                    while (active_thrust_1 == active_thrust_2):
                        active_thrust_2 = int(random.random() * self.numThrusts)

                    active_thrust_3 = active_thrust_2
                    while (active_thrust_2 == active_thrust_3):
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
# End of class
# ----------------------------


'''
Returns the proximity to a particular agent and
the index of the sensor that can sense the closest
part of the agent.
0 = not sensing anything; high value = high proximity
'''


def calculate_closest_agent_proximity(my_pos, other_pos, my_radius, other_radius):
    distance = np.linalg.norm(other_pos - my_pos)
    real_distance = max([distance - my_radius - other_radius, 0])

    proximity = get_proximity_linear(real_distance, my_radius)

    if proximity == 0:
        return None, 0

    xdiff = other_pos[0] - my_pos[0]
    ydiff = other_pos[1] - my_pos[1]
    angle_rad = np.arctan2(ydiff, xdiff)

    if angle_rad < 0:
        angle_rad += 2 * np.pi

    angle_deg = angle_rad * 180 / np.pi

    sensor_range_deg = 360 / c.INPUT_SENSOR_DIM  # with 4 sensors, the range is 90 degrees or pi/2

    # Determine the id of the active sensor, clockwise
    active_sensor = int(angle_deg / sensor_range_deg)

    # If the angle is exactly 360 degrees, this will return an invalid index
    if active_sensor == c.INPUT_SENSOR_DIM:
        active_sensor -= 1

    # print('sensoractivity',str(active_sensor))

    return active_sensor, proximity


def get_proximity_tanh(distance, my_radius):
    max_distance = c.MAX_DISTANCE * my_radius
    max_proximity = 1 / max_distance
    proximity = 1 / (distance + 1e-30)

    # Ignore sensor data when distance is too high
    if distance >= max_distance:
        return 0

    return (math.tanh(proximity - max_proximity) + 1e-30)


def get_proximity_linear(distance, my_radius):
    max_distance = c.MAX_DISTANCE * my_radius

    # Ignore sensor data when distance is too high
    if distance >= max_distance:
        return 0

    return - 1 * (distance / max_distance) + 1


def point_spread(sensor, proximity, functiontype):
    if functiontype not in ['linear', 'gauss', 'linear_normalized']:
        raise SystemExit("Point spread function unknown")

    sensor_data = np.zeros([c.INPUT_SENSOR_DIM])

    def linear(x):
        # f(x) = m*x + c
        # m is -1* SPREADSIZE
        # c is 1.0
        # So 1.0 is the maximum value
        return (-1 * c.SPREADSIZE) * abs(x) + 1.0

    def gauss(x):
        sigma = c.SIGMA
        # gauss = 1./np.sqrt(2*np.pi*sigma**2) * math.exp(-(abs(x)**2)/(2*sigma**2))
        gauss = math.exp(-(abs(x) ** 2) / (2 * sigma ** 2))
        return gauss

    if functiontype in ('linear', 'linear_normalized'):
        f = linear

    elif functiontype is 'gauss':
        f = gauss

    # Gehe von sensor aus die Indizes nach vorne (immer mit %)
    # Berechne fuer jeden Index i f(i).
    # Mache das solange nicht mehr als DIM/2 Schritte gegangen sind und solang f(i) > 0 ist
    steps = 1
    result = 1

    # Not normalized
    sensor_data[sensor] = f(0) * proximity

    while result > 0 and steps < c.INPUT_SENSOR_DIM / 2:
        result = f(steps) * proximity
        sensor_data[(sensor + steps) % c.INPUT_SENSOR_DIM] = result
        sensor_data[(sensor - steps) % c.INPUT_SENSOR_DIM] = result
        steps += 1

    if functiontype is 'linear_normalized':
        # Normalize the sensor data, so that sum is 1
        sensor_data /= np.sum(sensor_data)

    return sensor_data


'''
http://mathworld.wolfram.com/Circle-LineIntersection.html

'''


def calc_agent_intersection_proximity(my_pos, my_radius, other_pos, other_radius, sensor_vector):
    # The algorithm works for a circle at (0,0)
    # So translate the other_pos to (0,0)
    my_pos_translated = my_pos - other_pos
    other_pos_translated = other_pos - other_pos

    x_1 = my_pos_translated[0]  # start point of ray
    y_1 = my_pos_translated[1]  # start point of ray
    L = my_pos_translated + sensor_vector * 10  # end point of ray
    x_2 = L[0]
    y_2 = L[1]
    r = other_radius

    x_2_retranslated = x_2 + my_pos[0]
    y_2_retranslated = y_2 + my_pos[1]

    d_x = x_2 - x_1
    d_y = y_2 - y_1
    d_r = math.sqrt(d_x ** 2 + d_y ** 2)
    D = x_1 * y_2 - x_2 * y_1

    # fig, ax = plt.subplots()
    # ax.set_xlim(-100, 100)
    # ax.set_ylim(-100, 100)

    # Draw line
    # ax.plot([my_pos[0], x_2_retranslated], [my_pos[1], y_2_retranslated], '-')

    # Draw circle
    # circle = plt.Circle(other_pos, r)
    # ax.add_artist(circle)

    if (r ** 2 * d_r ** 2 - D ** 2) < 0:
        # plt.show()
        return None

    else:
        x_plus = (D * d_y + np.sign(d_y) * d_x * math.sqrt(r ** 2 * d_r ** 2 - D ** 2)) / d_r ** 2
        x_minus = (D * d_y - np.sign(d_y) * d_x * math.sqrt(r ** 2 * d_r ** 2 - D ** 2)) / d_r ** 2

        y_plus = (-D * d_x + abs(d_y) * math.sqrt(r ** 2 * d_r ** 2 - D ** 2)) / d_r ** 2
        y_minus = (-D * d_x - abs(d_y) * math.sqrt(r ** 2 * d_r ** 2 - D ** 2)) / d_r ** 2

        intersection1 = np.array([x_plus, y_plus])
        intersection2 = np.array([x_minus, y_minus])

        # determine closest of both intersection points
        dist1 = np.linalg.norm(intersection1 - my_pos_translated)
        dist2 = np.linalg.norm(intersection2 - my_pos_translated)

        if dist1 <= dist2:
            dist = dist1
            intersection = intersection1
        else:
            dist = dist2
            intersection = intersection2

        # prevent to detect intersections in wrong direction
        # if the distance of mypos to one of the intersection points gets bigger
        # when adding a dirs-Vector to mypos, then the intersection is in the wrong direction
        if ((intersection[0] <= x_2 and intersection[0] >= x_1) or (
                intersection[0] >= x_2 and intersection[0] <= x_1)) and \
                ((intersection[1] <= y_2 and intersection[1] >= y_1) or (
                        intersection[1] >= y_2 and intersection[1] <= y_1)):

            real_distance = dist - my_radius
            proximity = get_proximity_linear(real_distance, my_radius)
            return proximity

        else:
            return None


'''
https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
'''


def calc_border_intersection_proximity(my_pos, my_radius, border_pos, sensor_vector):
    x1 = my_pos[0]
    y1 = my_pos[1]

    L = my_pos + sensor_vector * 10  # end point of ray
    x2 = L[0]
    y2 = L[1]

    # plt.plot([x1, x2], [y1, y2], '-')

    x3 = border_pos[0, 0]
    y3 = border_pos[0, 1]
    x4 = border_pos[1, 0]
    y4 = border_pos[1, 1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    P_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    P_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    # Check if the intersection is in the direction of the sensor vector
    # If the intersection point is between mypos and L, it's cool
    if ((P_x <= x2 and P_x >= x1) or (P_x >= x2 and P_x <= x1)) and \
            ((P_y <= y2 and P_y >= y1) or (P_y >= y2 and P_y <= y1)):

        intersection = np.array([P_x, P_y])
        distance = np.linalg.norm(intersection - my_pos)

        # plt.plot(intersection[0], intersection[1], 'r*')
        # plt.show()

        real_distance = distance - my_radius
        proximity = get_proximity_linear(real_distance, my_radius)
        return proximity * c.BORDER_PROXIMITY_WEIGHT

    else:
        # plt.show()
        return None


def calc_sensor_dirs():
    dirs = []
    for i in range(c.INPUT_SENSOR_DIM):
        sensor_range_rad = (2 * np.pi) / c.INPUT_SENSOR_DIM
        angle_rad = (i * sensor_range_rad + (i + 1) * sensor_range_rad) / 2.
        dirs.append(np.array([np.cos(angle_rad), np.sin(angle_rad)]))

    dirs = np.asarray(dirs)
    return dirs


class Simulator_State(object):
    def __init__(self, current_pos, previous_pos, current_velocity):
        self.velocity = current_velocity

        self.current_position = np.zeros(2)
        self.current_position[0] = current_pos[0]
        self.current_position[1] = current_pos[1]

        self.previous_position = np.zeros(2)
        self.previous_position[0] = previous_pos[0]
        self.previous_position[1] = previous_pos[1]

