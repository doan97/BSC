import numpy as np
import random
from ringbuffer import RingBuffer
from simulator import Simulator
import global_config as c
import math


class Data(object):
    def __init__(self, sim):
        self.reset()
        self.sim = sim

    def reset(self, new_idx=None):
        # All data sets represent the data **before** executing the time step
        # positions[10] represents the position at time step 10, before executing
        # motorcommands[10] and moving to positions[11]
        # Same for sensors. sensors[10] represents the sensors after time step 9 and before executing step 10,
        # so the perception of all agents.positions[10] (which are the positions after step 9 and before 10
        self.positions = RingBuffer(c.RINGBUFFER_SIZE, c.OUTPUT_POSITION_DIM)
        self.position_deltas = RingBuffer(c.RINGBUFFER_SIZE, c.OUTPUT_POSITION_DIM)
        self.motor_commands = RingBuffer(c.RINGBUFFER_SIZE, c.INPUT_MOTOR_DIM)
        self.sensors = RingBuffer(c.RINGBUFFER_SIZE, c.INPUT_SENSOR_DIM)
        self.accelerations = RingBuffer(c.RINGBUFFER_SIZE, 2)

        # For actinf
        self.states = RingBuffer(c.RINGBUFFER_SIZE, [2, 1, c.HIDDEN_DIM])
        self.actinf_targets = RingBuffer(c.RINGBUFFER_SIZE, c.OUTPUT_POSITION_DIM)

        self.scv = RingBuffer(c.RINGBUFFER_SIZE, c.INPUT_POSITION_DIM)

        # For simulator
        self.velocities = RingBuffer(c.RINGBUFFER_SIZE, c.INPUT_POSITION_DIM)

        if new_idx is not None:
            self.accelerations.change_curr_idx(new_idx + 1)
            self.position_deltas.change_curr_idx(new_idx + 1)
            self.positions.change_curr_idx(new_idx + 1)
            self.states.change_curr_idx(new_idx + 1)
            self.velocities.change_curr_idx(new_idx + 1)
            self.actinf_targets.change_curr_idx(new_idx)

            # TODO
            # sensors
            # scv

    def get_combined_inputs_block(self, from_index, to_index):

        v = self.position_deltas.get(from_index, to_index)
        m = self.motor_commands.get(from_index, to_index)

        # Velocity Dropout 33%
        if False:
            for vp in range(len(v)):
                if vp % 3 == 0:
                    v[vp][0] = 0
                    v[vp][1] = 0

        combined_inputs = np.concatenate([v, m], axis=1)

        if c.INPUT_SENSOR_DIM > 0:
            s = self.sensors.get(from_index, to_index)
            combined_inputs = np.concatenate([combined_inputs, s], axis=1)

        if c.INPUT_ACCELERATION_DIM > 0:
            a = self.accelerations.get(from_index, to_index)
            combined_inputs = np.concatenate([combined_inputs, a], axis=1)

        # combined inputs = [velocity, motor_commands, sensor_data, accelerations]
        return combined_inputs

    def create_inputs(self, num_time_steps, current_agent, stand_still=False):
        for i in range(0, num_time_steps):

            # Get the last motor command
            # Always access relative with -1, because curr_idx of motor_commands grows at 'append_single' below
            motor_commands = self.motor_commands.get_relative(-1)

            # Execute the last motor command to get the velocity and position
            previous_velocity = self.velocities.get_relative(-1)
            previous_position = self.positions.get_relative(-1)
            position_delta, position, velocity, acceleration = self.sim.update(previous_velocity, previous_position,
                                                                               motor_commands, current_agent,
                                                                               self.positions.curr_idx - 1)


            #these are the targets
            self.position_deltas.append_single(position_delta)
            self.positions.append_single(position)
            self.velocities.append_single(velocity)
            self.accelerations.append_single(acceleration)

            # Create the next motorcommand

            # if already moved
            if stand_still is False:
                next_motor_commands = self.sim.get_random_motor_commands(last_commands=motor_commands)

            # if have not moved
            else:
                next_motor_commands = np.zeros_like(motor_commands)

            self.motor_commands.append_single(next_motor_commands)

        # endfor

    def get_targets_block(self, from_index, to_index):
        """ returns the targets of the RNN.
        :param from_index:
        :param to_index:
        :return: targets = [position_deltas, sensor_targets, acceleration_targets]
        """
        # The position of time step t is the target for time step t-1
        real_from_index = from_index + 1
        real_to_index = to_index + 1

        targets = self.position_deltas.get(real_from_index, real_to_index)

        if c.OUTPUT_SENSOR_DIM > 0:
            # In time step t, the sensor input of time step t+1 must be predicted
            sensor_targets = self.sensors.get(real_from_index, real_to_index)
            targets = np.concatenate([targets, sensor_targets], axis=1)

        if c.OUTPUT_ACCELERATION_DIM > 0:
            acceleration_targets = self.accelerations.get(real_from_index, real_to_index)
            targets = np.concatenate([targets, acceleration_targets], axis=1)
        # inputs= [velocity, motor_commands, sensor_data, accelerations]
        # outputs = [position_deltas, sensor_inputs, acceleration]
        # targets = [position_deltas, sensor_targets, acceleration_targets]
        return targets

    def get_actinf_targets_block(self, from_index, to_index):
        targets = self.actinf_targets.get(from_index, to_index)
        return targets

    '''
    Returns the target coordinates, just for information
    additional_time_steps: We need to create the targets in the future to avoid following 0-targets in prospective inference
    '''

    def create_actinf_targets(self, time_steps_follow, additional_time_steps):

        radius = 0.06

        start_x = -1.5 + radius
        end_x = 1.5 - radius
        x_range = end_x - start_x

        start_y = 0 + 0.06
        end_y = 2 - 0.06
        y_range = end_y - start_y

        # create new random target
        # x between -1.5 and 1.5
        target_x = (random.random() * x_range) + start_x

        # y between 0 and 2
        target_y = random.random() * y_range + start_y

        target = np.array([target_x, target_y])

        target = target[np.newaxis, :]
        targets = np.repeat(target, [time_steps_follow + additional_time_steps], axis=0)

        self.actinf_targets.append(targets)
        self.actinf_targets.change_curr_idx(-1 * additional_time_steps)

        return target[-1, :]

    '''
    Returns the target coordinates, just for information
    additional_time_steps: We need to create the targets in the future to avoid following 0-targets in prospective inference
    '''

    def create_actinf_targets_line(self, time_steps_follow, additional_time_steps):

        targets = []

        if random.random() < 0.5:
            # vertical line
            target_x = (random.random() * 3) - 1.5
            target_y = 0

            target = np.array([target_x, target_y])
            targets.append(target)

            # Now create targets on a line from the created one
            num_steps = time_steps_follow + additional_time_steps
            step_size = 2. / num_steps

            for step in range(num_steps - 1):
                next_target = np.copy(target)
                next_target[1] += step_size * step
                targets.append(next_target)

        else:  # horizontal line
            target_x = -1.5
            target_y = random.random() * 2

            target = np.array([target_x, target_y])
            targets.append(target)

            # Now create targets on a line from the created one
            num_steps = time_steps_follow + additional_time_steps
            step_size = 3. / num_steps

            for step in range(num_steps - 1):
                next_target = np.copy(target)
                next_target[0] += step_size * step
                targets.append(next_target)

        targets = np.array(targets)

        self.actinf_targets.append(targets)
        self.actinf_targets.change_curr_idx(-1 * additional_time_steps)

        return targets

    def set_actinf_targets(self, target, time_steps_follow, additional_time_steps):
        target = target[np.newaxis, :]

        targets = np.repeat(target, [time_steps_follow + additional_time_steps], axis=0)

        self.actinf_targets.append(targets)
        self.actinf_targets.change_curr_idx(-1 * additional_time_steps)

        return target[-1, :]
