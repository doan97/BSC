import WORKING_MARCO.global_config as c
import numpy as np
import torch

from WORKING_MARCO.Net import Net
from WORKING_MARCO.agent import Agent
from WORKING_MARCO.simulator import Simulator
from WORKING_MARCO.gui import GUI
from WORKING_MARCO.stopwatch import Stopwatch
import WORKING_MARCO.learn as learn
from tkinter import *
import torch.nn as nn

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

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

def transform_pos(pos, min_x, max_x, min_y, max_y, new_width, new_height):
    x = pos
    old_width = max_x - min_x #3
    old_height = max_y - min_y #2
    new_x = [x[0] - min_x, np.abs(x[1]  - max_y)]
    new_x = [new_width / old_width * new_x[0], new_height/ old_height * new_x[1]]
    return new_x

def get_sensor_data(agent, from_step, input_t, current_position):
    sensor_data = agent.sim.calc_sensor_data(from_step + input_t, from_step + input_t + 1, agent,
                                             agent.data.sim.sensor_dirs, agent.data.sim.borders, current_position)
    return sensor_data

def simulate(simulator, agent, com, previous_velocity, previous_position, from_step, input_t):
    sim_position_delta, sim_position, new_velocity, sim_acceleration = \
        simulator.update(previous_velocity, previous_position, com, agent, from_step + input_t)
    agent.data.positions.append_single(sim_position)

    return sim_position_delta, sim_position, new_velocity, sim_acceleration

def predict(coms, agent, position_delta, sensor_data, acceleration, input_type='all'):
    position_delta_tensors = []
    acceleration_tensors = []
    sensor_tensors = []
    position_delta_predictions =[position_delta]
    acceleration_predictions = [acceleration]
    sensor_predictions = [sensor_data]
    inputs = []

    hidden = [(torch.zeros(1, 1, 36), torch.zeros(1, 1, 36))]

    for idx, com in enumerate(coms):

        if idx == 0:
            if input_type == 'motor only':
                input_data = com
            elif input_type == 'motor and sensor':
                input_data = np.concatenate([com, sensor_predictions[-1]],axis=1)
            elif input_type == 'all':
                input_data = np.concatenate([position_delta_predictions[-1], com], axis=1)
                input_data = np.concatenate([input_data, sensor_predictions[-1]], axis=1)
                input_data = np.concatenate([input_data, acceleration_predictions[-1]], axis=1)
            else:
                print('input type is not implemented')
                return
            input_data = input_data[:, np.newaxis, :]
            input_data = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)

        inputs.append(input_data)
        pred, h = agent.net.forward(input_data, hidden_=hidden[-1])
        pred = pred.view(-1)
        hidden.append(h)
        position_delta_tensors.append(pred[:2])
        acceleration_tensors.append(pred[18:])
        sensor_tensors.append(pred[2:18])

        position_delta_predictions.append(pred[0:2])
        acceleration_predictions.append(pred[18:20])
        sensor_predictions.append(pred[2:18])
    return position_delta_tensors, sensor_tensors, inputs



def action_inference_step(position_delta_predictions, position_delta_targets, sensor_predictions, sensor_targets,inputs, learning_rate=0.01, input_type='all', seek_proximity=False):

    mse = nn.MSELoss()
    predictions_delta_position = torch.stack(position_delta_predictions)
    position_delta_targets = torch.tensor(position_delta_targets, dtype=torch.float32)
    predictions_sensor = torch.stack(sensor_predictions)

    if seek_proximity:
        sensor_loss = mse(predictions_sensor, sensor_targets)
        inputs_clone = [input for input in inputs]
        optimized_velocities = optimize_velocities(input_type,inputs_clone, learning_rate, sensor_loss)
        position_loss = mse(predictions_delta_position, optimized_velocities)
    else:
        position_loss = mse(predictions_delta_position, position_delta_targets)
    sensor_loss = mse(predictions_sensor, sensor_targets)

    optimized_motor_commands = optimize_motor_commands_vel_induced(input_type, inputs, learning_rate, position_loss, sensor_loss, seek_proximity=seek_proximity)

    return optimized_motor_commands

def optimize_velocities(input_type, inputs, learning_rate, sensor_loss, sens_l=100):
    sensor_loss = sens_l * sensor_loss
    optimizer = torch.optim.Adam(inputs, lr=learning_rate)
    optimizer.zero_grad()

    sensor_loss.backward(retain_graph=True)
    mask_gradients_(inputs, input_type=input_type, velinf=True)
    optimizer.step()
    inputs_tensor = torch.stack(inputs)
    optimized_velocities = inputs_tensor.data.numpy()[:, -1, -1, :2]
    optimized_velocities = np.clip(optimized_velocities, 0, 1)
    return torch.tensor(optimized_velocities, dtype=torch.float32)



def optimize_motor_commands_vel_induced(input_type, inputs, learning_rate, position_loss, sensor_loss, vel_l=100, sens_l=100, velinf=False, seek_proximity=False):
    velocity_loss = vel_l * position_loss
    loss = velocity_loss
    optimizer = torch.optim.Adam(inputs, lr=learning_rate)
    optimizer.zero_grad()
    loss.backward()
    if input_type == 'all':
        mask_gradients_(inputs, input_type=input_type)
    else:
        mask_gradients_(inputs, input_type=input_type)
    optimizer.step()
    inputs_tensor = torch.stack(inputs)
    if input_type == 'all':
        optimized_motor_commands = inputs_tensor.data.numpy()[:, -1, -1, c.MOTOR_DIM_START:c.MOTOR_DIM_END]
    elif input_type == 'motor and sensor' or input_type == 'motor only':
        optimized_motor_commands = inputs_tensor.data.numpy()[:, -1, -1, :4]
    optimized_motor_commands = np.clip(optimized_motor_commands, 0, 1)
    return optimized_motor_commands


def get_velocity_delta_targets(init_pos, pos_delta_pred, position_targets):
    abs_position_predictions = torch.tensor(init_pos, dtype=torch.float) + torch.cumsum(
        torch.stack(pos_delta_pred), dim=0)
    return position_targets - abs_position_predictions.data.numpy()

def get_sensor_targets(sensor_predictions):
    sensor_predictions = torch.stack(sensor_predictions)
    sensor_targets = np.zeros_like(sensor_predictions.data.numpy())

    for t in range(len(sensor_targets)):
            max_i = 0
            max_val = 0
            for i in range(len(sensor_predictions[t])):
                if sensor_predictions[t, i].item() > max_val:
                    max_val = sensor_predictions[t, i].item()
                    max_i = i

            sensor_targets[t, max_i] = 1.

    for t in range(len(sensor_targets)):
            # Apply Gauss to sensor_targets
            point_spread_signal = np.zeros([c.INPUT_SENSOR_DIM])

            for sensor_index in range(c.INPUT_SENSOR_DIM):
                point_spread_signal_tmp = point_spread(sensor_index, sensor_targets[t, sensor_index],
                                                       c.POINT_SPREAD_TYPE)
                point_spread_signal_tmp[sensor_index] = 0
                point_spread_signal += point_spread_signal_tmp
            # Only after all point spread additions is calculated, add it to the sensor data
            sensor_targets[t] = sensor_targets[t] + point_spread_signal
            sensor_targets[t] = np.clip(sensor_targets[t], 0, 1)
            sensor_targets[t] = np.round(sensor_targets[t], decimals=8)

    sensor_targets = torch.tensor(sensor_targets)
    return sensor_targets

def actinf_iteration(agent, init_pos, targets_pos, velocity, acceleration, sensor_information, prediction_horizon, input_type='all', seek_proximity=False):
    coms = np.random.random((prediction_horizon,4))[:, np.newaxis]
    position_delta_predictions, sensor_predictions, inputs = predict(coms, agent, velocity, sensor_information, acceleration, input_type=input_type)
    position_delta_targets = get_velocity_delta_targets(init_pos, position_delta_predictions, targets_pos)
    sensor_targets = get_sensor_targets(sensor_predictions)
    motor_commands = action_inference_step(position_delta_predictions, position_delta_targets, sensor_predictions, sensor_targets,inputs, input_type=input_type, seek_proximity=seek_proximity)
    return motor_commands

def action_inference(model_file, init_pos, target_position, obstacle_positions, timesteps=1, actinf_iterations=1, prediction_horizon=1, input_type='all', seek_proximity=False):
    gui = GUI()
    s = Stopwatch()
    sim = Simulator(mode=1, stopwatch=s)
    agent = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                   model_file=model_file, input_type=input_type)
    obstacles = learn.create_obstacles(len(obstacle_positions), gui=gui, sim=sim, \
                                       positions=np.array(obstacle_positions))
    agent.register_agents(obstacles)
    positions = []
    sensor_information_of_all_steps = []
    current_position = init_pos
    current_velocity = np.array([0,0])
    current_position_delta = np.array([0,0])
    current_acceleration = np.array([0,0])
    for t in range(timesteps):
        for ai in range(actinf_iterations):
            current_sensor_information = get_sensor_data(agent, t, 0, current_position)
            sensor_information_of_all_steps.append(current_sensor_information)
            motor_commands = actinf_iteration(agent,
                                             np.array([current_position]),
                                             np.array([target_position]),
                                             np.array([current_position_delta]),
                                             np.array([current_acceleration]),
                                             current_sensor_information,
                                             prediction_horizon,\
                                              input_type=input_type,
                                              seek_proximity=seek_proximity)
        next_motor_command = motor_commands[0]

        current_position_delta, current_position, current_velocity, current_acceleration = \
            simulate(sim, agent, next_motor_command, current_velocity, current_position, t, 0)
        agent.data.positions.append_single(current_position)

        positions.append(current_position)

    return positions, sensor_information_of_all_steps

def draw_path(positions, targets, obstacles):
    master = Tk()

    w = Canvas(master, width=1000, height=1000)
    w.pack()
    # draw border
    w.create_rectangle(0, 0, 600, 400)
    # draw_obstacles
    for p in targets:
        p = transform_pos(p, -1.5, 1.5, 0, 2, 600, 400)
        r = 10
        w.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, fill='yellow')

    for p in obstacles:
        p = transform_pos(p, -1.5, 1.5, 0, 2, 600, 400)
        r = 10
        w.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, fill='red')

        # draw real positions
    for idx, pos in enumerate(positions):
        r = 2
        if idx + 1 < len(positions):
            pos = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            next_pos = positions[idx + 1]
            next_pos = transform_pos(next_pos, -1.5, 1.5, 0, 2, 600, 400)
            w.create_line(pos[0], pos[1], next_pos[0], next_pos[1], fill='cornflowerblue', width=3)
            # w.create_oval(pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r, fill='black')
    mainloop()

def draw_sensor_data(sensor_data):
    master = Tk()

    w = Canvas(master, width=1000, height=1000)
    w.pack()
    r= 10
    p = np.array([100,100])
    for idx, d in enumerate(sensor_data):
        #np.sin()
        w.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, fill='yellow')
        angle_stepper = 0
        for angle_value in d[0]:
            angle_step = 360 / len(d[0])
            y = round(np.sin(np.deg2rad(angle_stepper)), 2)
            angle_stepper += angle_step
            x = angle_value - y**2
            w.create_line(p[0], p[1], int(p[0] + x * 50), int(p[1] + y*50), fill='blue')
        p[0] += 100
        if p[0] == 1000:
            p[0] = 0
            p[1] += 100
        print(p)
    mainloop()

def draw_timestep(sensor_data, positions, obstacles, timestep):
    master = Tk()

    w = Canvas(master, width=1200, height=1000)
    w.pack()
    r = 5

    sensor = sensor_data[timestep][0]
    position = transform_pos(positions[timestep], -1.5, 1.5, 0, 2, 600, 400)
    w.create_oval(position[0] - r, position[1] - r, position[0] + r, position[1] + r, fill='yellow')
    #draw obstacles
    for o in obstacles:
        p = transform_pos(o, -1.5, 1.5, 0, 2, 600, 400)
        r = 10
        w.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, fill='red')
    angle_stepper = 0
    for value in sensor:
        angle_step = 360 / len(sensor)
        y = round(np.sin(np.deg2rad(angle_stepper)), 2)
        angle_stepper += angle_step
        x = value - y**2
        w.create_line(position[0], position[1], int(position[0] + x * 50), int(position[1] + y*50), fill='blue')

    mainloop()

def run_that_shit(init_pos, target, obstacles):
    pos, sensor_data = action_inference('model_A_border', np.array(init_pos), np.array(target), obstacles,
                           timesteps=50, actinf_iterations=10,
                           prediction_horizon=10, input_type='all', seek_proximity=True)

    #draw_path(pos, [target], obstacles)
    #draw_sensor_data(sensor_data)
    draw_timestep(sensor_data, pos, obstacles, 20)



run_that_shit([0,1],[-1, 1.8], [[-0.2,1.5], [0.2,1.5], [0,0.5]])