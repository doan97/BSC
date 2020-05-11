import numpy as np
import bachelor_vinhdo.global_config as c
import torch
import random
random.seed(1)

def get_command_sequence(count, start_command):
    sequence = [start_command]
    for i in range(count - 1):
        sequence.append(get_random_motor_commands(start_command))
    return np.array(sequence)[:, np.newaxis]

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

def mask_gradients_(inputs, input_type='all', velinf=False):
    if input_type == 'all':
        if velinf:
            mask = np.concatenate([
                np.ones(c.INPUT_POSITION_DIM, dtype=np.float32),
                np.zeros(c.INPUT_MOTOR_DIM, dtype=np.float32),
                np.zeros(c.INPUT_SENSOR_DIM, dtype=np.float32),
                np.zeros(c.INPUT_ACCELERATION_DIM, dtype=np.float32)
            ], axis=0)
        else:
            mask = np.concatenate([
                np.zeros(c.INPUT_POSITION_DIM, dtype=np.float32),
                np.ones(c.INPUT_MOTOR_DIM, dtype=np.float32),
                np.zeros(c.INPUT_SENSOR_DIM, dtype=np.float32),
                np.zeros(c.INPUT_ACCELERATION_DIM, dtype=np.float32)
            ], axis=0)
    elif input_type == 'motor and sensor':
        mask = np.concatenate([
            np.ones(c.INPUT_MOTOR_DIM, dtype=np.float32),
            np.zeros(c.INPUT_SENSOR_DIM, dtype=np.float32),
        ], axis=0)
    elif input_type == 'motor only':
        return None

    mask = torch.tensor(mask)
    for i in inputs:
        i.grad = i.grad * mask

def get_sensor_data(agent, from_step, input_t, current_position, obstacles):
    sensor_data = agent.sim.calc_sensor_data(from_step + input_t, from_step + input_t + 1, agent,
                                             agent.data.sim.sensor_dirs, agent.data.sim.borders, current_position, obstacles)
    return sensor_data

def simulate(simulator, agent, com, previous_velocity, previous_position, from_step, input_t, obstacles):
    sim_position_delta, sim_position, new_velocity, sim_acceleration = \
        simulator.update(previous_velocity, previous_position, com, agent, from_step + input_t, obstacles)
    agent.data.positions.append_single(sim_position)

    return sim_position_delta, sim_position, new_velocity, sim_acceleration

def get_random_motor_commands(last_commands):
    motor_commands = np.zeros(4)

    r = random.random()

    if r < 0.7:  # 70%: new commands

        r = random.random()

        # 0 of 4 active .2
        # 1 of 4 active .2
        # 2 of 4 active .2
        # 3 of 4 active .2
        # 4 of 4 active .2

        if r < 0.2:  # 20%: do nothing
            motor_commands = np.zeros(4)

        elif r < 0.4:  # 40%: 1 active
            active_thrust = int(random.random() * 4)
            motor_commands[active_thrust] = random.random()

        elif r < 0.6:  # 20%: 2 active
            active_thrust_1 = int(random.random() * 4)

            active_thrust_2 = active_thrust_1
            while active_thrust_1 == active_thrust_2:
                active_thrust_2 = int(random.random() * 4)

            motor_commands[active_thrust_1] = random.random()
            motor_commands[active_thrust_2] = random.random()

        elif r < 0.8:  # 20%: 3 active
            active_thrust_1 = int(random.random() * 4)

            active_thrust_2 = active_thrust_1
            while active_thrust_1 == active_thrust_2:
                active_thrust_2 = int(random.random() * 4)

            active_thrust_3 = active_thrust_2
            while active_thrust_2 == active_thrust_3:
                active_thrust_3 = int(random.random() * 4)

            motor_commands[active_thrust_1] = random.random()
            motor_commands[active_thrust_2] = random.random()
            motor_commands[active_thrust_3] = random.random()

        else:  # 20%: all active
            motor_commands = np.random.random(size=4)

    else:
        motor_commands = last_commands

    return motor_commands
