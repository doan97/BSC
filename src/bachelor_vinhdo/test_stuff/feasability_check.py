from random import random
import bachelor_vinhdo.global_config as c
import numpy as np
import torch

from WORKING_MARCO.Net import Net
from bachelor_vinhdo.agent import Agent
from bachelor_vinhdo.simulator import Simulator
from WORKING_MARCO.gui import GUI
from WORKING_MARCO.stopwatch import Stopwatch
import bachelor_vinhdo.learn as learn
import bachelor_vinhdo.utils as utils
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

NUM_TIME_STEPS = 150



#simulate real data
def simulation(init_pos, commands, sim, gui, obstacle_positions):
    agent = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                  model_file='../models/from_tests_before/model_A', less_inputs=True)
    #obstacles = learn.create_obstacles(4, gui=gui, sim=sim, \
    #                                   positions=np.array(obstacle_positions))
    #agent.register_agents(obstacles)
    positions = [init_pos]

    velocities = [np.array([0,0])]
    obstacles = []
    for p in obstacle_positions:
        obstacles.append({'position': p, \
                          'radius': 0.06})
    for idx, com in enumerate(commands):
        sim_position_delta, sim_position, new_velocity, sim_acceleration = \
            sim.update(velocities[-1], positions[-1], com, agent, idx, obstacles, borders=True)
        positions.append(sim_position)
        velocities.append(new_velocity)
    return positions


def lstm_prediction_sim_all(init_pos, commands, agent, sim, obstacles, position_info=True, motor_only=False):
    positions = [init_pos]
    velocities = [np.array([0,0])]
    accelerations = [np.array([0,0])]
    position_deltas = [np.array([0,0])]

    position_predictions = []
    velocity_predictions = []

    hidden = [(torch.zeros(1, 1, 36), torch.zeros(1, 1, 36))]
    for idx, com in enumerate(commands):
        #generate input
        sensor_data = sim.calc_sensor_data(idx, idx + 1, agent,
                                           agent.data.sim.sensor_dirs,
                                           agent.data.sim.borders, positions[-1], obstacles)[0]
        if position_info:
            inputs = [[np.concatenate((position_deltas[-1], com ,sensor_data,  accelerations[-1]), axis=0)]]
        else:
            inputs = [[np.concatenate((com ,sensor_data), axis=0)]]
        if motor_only:
            inputs = [[com]]
        inputs = torch.tensor(inputs, dtype=torch.float32)
        prediction, new_hidden = agent.net.forward(inputs, hidden_=hidden[-1])
        hidden.append(new_hidden)

        position_predictions.append(positions[-1] + prediction[:, :2].detach().numpy()[0])
        velocity_predictions.append(velocities[-1] + prediction[:, :2].detach().numpy()[0])

        #calc next real inputs
        sim_position_delta, sim_position, new_velocity, sim_acceleration = \
            sim.update(velocities[-1], positions[-1], com, agent, idx, obstacles)

        positions.append(sim_position)
        velocities.append(new_velocity)
        accelerations.append(sim_acceleration)
        position_deltas.append(sim_position_delta)

    return position_predictions

def lstm_prediction_sim_sensor(init_pos, commands, agent, sim, obstacles, position_info=True, loop_start=0, motor_only=False):
    positions = [init_pos]
    velocities = [np.array([0,0])]
    accelerations = [np.array([0,0])]

    position_predictions = [init_pos]
    velocity_predictions = [np.array([0,0])]
    acceleration_predictions = [np.array([0,0])]
    hidden = [(torch.zeros(1, 1, 36), torch.zeros(1, 1, 36))]
    for idx, com in enumerate(commands):
        if idx < loop_start:
            sensor_data = sim.calc_sensor_data(idx, idx + 1, agent,
                                               agent.data.sim.sensor_dirs,
                                               agent.data.sim.borders, positions[-1], obstacles)[0]
            sim_position_delta, sim_position, new_velocity, sim_acceleration = \
                sim.update(velocities[-1], positions[-1], com, agent, idx, obstacles)

            if position_info:
                inputs = [[np.concatenate((velocities[-1], com, sensor_data, accelerations[-1]), axis=0)]]
            else:
                inputs = [[np.concatenate((com, sensor_data), axis=0)]]
            if motor_only:
                inputs = [[com]]
            inputs = torch.tensor(inputs, dtype=torch.float32)
            prediction, (h, c) = agent.net.forward(inputs, hidden_=hidden[-1])
            hidden.append((h.detach(), c.detach()))

            position_predictions.append(sim_position)
            velocity_predictions.append(sim_position_delta)
            acceleration_predictions.append(sim_acceleration)

            positions.append(sim_position)
            velocities.append(sim_position_delta)
            accelerations.append(sim_acceleration)
        else:
            sensor_data = sim.calc_sensor_data(idx, idx + 1, agent,
                                               agent.data.sim.sensor_dirs,
                                               agent.data.sim.borders, positions[-1], obstacles)[0]
            sim_position_delta, sim_position, new_velocity, sim_acceleration = \
                sim.update(velocities[-1], positions[-1], com, agent, idx, obstacles)

            positions.append(sim_position)
            velocities.append(sim_position_delta)

            if position_info:
                inputs = [[np.concatenate((velocity_predictions[-1], com ,sensor_data,  acceleration_predictions[-1]), axis=0)]]
            else:
                inputs = [[np.concatenate((com ,sensor_data), axis=0)]]
            if motor_only:
                inputs = [[com]]
            inputs = torch.tensor(inputs, dtype=torch.float32)
            prediction, (h, c) = agent.predict(inputs, hidden[-1])
            hidden.append((h.detach(), c.detach()))

            acceleration_prediction = prediction[:, -2:].detach().numpy()[0]
            velocity_prediction = prediction[:, :2].detach().numpy()[0]
            position_prediction = position_predictions[-1] + velocity_prediction

            acceleration_predictions.append(acceleration_prediction)
            velocity_predictions.append(velocity_prediction)
            position_predictions.append(position_prediction)

    return position_predictions

def lstm_prediction_loop(init_pos, commands, agent, sim, obstacles,position_info=True, loop_start=10, motor_only=False):
    position_predictions = [init_pos]
    velocity_predictions = [np.array([0,0])]
    acceleration_predictions = [np.array([0,0])]
    sensor_predictions = []

    sensor_datas = []
    #first sensor input is calculated by simulator

    hidden = [(torch.zeros(1, 1, 36), torch.zeros(1, 1, 36))]


    for idx, com in enumerate(commands):
        if idx < loop_start:
            sensor_data = sim.calc_sensor_data(idx, idx + 1, agent,
                                               agent.data.sim.sensor_dirs,
                                               agent.data.sim.borders, init_pos, obstacles)[0]
            sim_position_delta, sim_position, new_velocity, sim_acceleration = \
                sim.update(velocity_predictions[-1], position_predictions[-1], com, agent, idx, obstacles)

            if position_info:
                inputs = [[np.concatenate((velocity_predictions[-1], com ,sensor_data,  acceleration_predictions[-1]), axis=0)]]
            else:
                inputs = [[np.concatenate((com ,sensor_data), axis=0)]]
            if motor_only:
                inputs = [[com]]
            inputs = torch.tensor(inputs, dtype=torch.float32)
            prediction, (h, c) = agent.predict(inputs, hidden[-1])
            hidden.append((h.detach(), c.detach()))

            #add real data so the model can start with right inputs
            position_predictions.append(sim_position)
            velocity_predictions.append(sim_position_delta)
            acceleration_predictions.append(sim_acceleration)
            sensor_predictions.append(prediction[:, 2:18].detach().numpy()[0])

        else:
            if position_info:
                inputs = [[np.concatenate((velocity_predictions[-1], com ,sensor_predictions[-1],  acceleration_predictions[-1]), axis=0)]]
            else:
                inputs = [[np.concatenate((com ,sensor_predictions[-1]), axis=0)]]
            if motor_only:
                inputs = [[com]]

            inputs = torch.tensor(inputs, dtype=torch.float32)
            prediction, new_hidden = agent.predict(inputs, hidden[-1])
            hidden.append(new_hidden)

            acceleration_prediction = prediction[:, -2:].detach().numpy()[0]
            position_delta_prediction = prediction[:, :2].detach().numpy()[0]

            acceleration_predictions.append(acceleration_prediction)
            position_predictions.append(position_predictions[-1] + position_delta_prediction)
            velocity_predictions.append(position_delta_prediction)#velocity_predictions[-1] + acceleration_prediction)
            sensor_predictions.append(prediction[:, 2:18].detach().numpy()[0])
    return position_predictions

def sim_all(init_pos, sim, gui, commands, obstacle_positions):
    agent_A = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                    model_file='../models/from_tests_before/model_A', input_type='all')
    agent_B = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                    model_file='../models/from_tests_before/model_B', input_type='motor and sensor')
    agent_C = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                    model_file='../models/from_tests_before/model_C', input_type='motor only')
    obstacles = []
    for p in obstacle_positions:
        obstacles.append({'position' : p,\
                          'radius' : 0.06})

    p_A = lstm_prediction_sim_all(init_pos, commands, agent_A, sim, obstacles,position_info=True)
    p_B = lstm_prediction_sim_all(init_pos, commands, agent_B, sim, obstacles,position_info=False)
    p_C = lstm_prediction_sim_all(init_pos, commands, agent_C, sim, obstacles, motor_only=True)
    return p_A, p_B, p_C

def sim_sens(init_pos, sim, gui, commands, obstacle_positions):
    agent_A = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                    model_file='../models/from_tests_before/model_A')
    agent_B = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                    model_file='../models/from_tests_before/model_B', input_type='motor and sensor')
    agent_C = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                    model_file='../models/from_tests_before/model_C', input_type='motor only')
    obstacles = []
    for p in obstacle_positions:
        obstacles.append({'position': p, \
                          'radius': 0.06})

    p_A = lstm_prediction_sim_sensor(init_pos, commands, agent_A, sim, obstacles,position_info=True, loop_start=10)
    p_B = lstm_prediction_sim_sensor(init_pos, commands, agent_B, sim, obstacles,position_info=False, loop_start=10)
    p_C = lstm_prediction_sim_sensor(init_pos, commands, agent_C, sim, obstacles,motor_only=True, loop_start=10)
    return p_A, p_B, p_C

def loop(init_pos, sim, gui, commands, obstacle_positions, loop_start=10):
    agent_A = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                    model_file='../models/from_tests_before/model_A')
    agent_B = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                    model_file='../models/from_tests_before/model_B', input_type='motor and sensor')
    agent_C = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                    model_file='../models/from_tests_before/model_C', input_type='motor only')
    obstacles = learn.create_obstacles(4, gui=gui, sim=sim, \
                                       positions=np.array(obstacle_positions))
    agent_A.register_agents(obstacles)
    agent_B.register_agents(obstacles)
    agent_C.register_agents(obstacles)

    obstacles = []
    for p in obstacle_positions:
        obstacles.append({'position': p, \
                          'radius': 0.06})

    p_A = lstm_prediction_loop(init_pos, commands, agent_A, sim, obstacles, position_info=True,
                                                    loop_start=loop_start)
    p_B = lstm_prediction_loop(init_pos, commands, agent_B, sim, obstacles, position_info=False,
                                                    loop_start=loop_start)
    p_C = lstm_prediction_loop(init_pos, commands, agent_C, sim, obstacles, motor_only=True,
                               loop_start=loop_start)
    return p_A, p_B, p_C


def get_all_positions_data():
    gui = GUI()
    s = Stopwatch()
    sim = Simulator(mode=1, stopwatch=s)
    start_position = np.array([0, 1.])
    #commands = np.load('feasabilty_coms.npy')
    commands = [np.array([1, 0, 1, 0]) for i in range(100)]
    #commands = curve_motor_coms(100, 2,3)
    #commands = generate_square_motor_commands()
    commands = utils.get_command_sequence(100, np.zeros(4))
    commands = [command[0] for command in commands]
    obstacle_positions = [[-1, 1.5], [1., 1.5], [0, 1.5], [1.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [0., 0.5], [1.0, 0.5], [-1.0, 0.5]]

    real_positions = simulation(start_position, commands, sim, gui, obstacle_positions)

    pred_positions_sim_all_A, pred_positions_sim_all_B, pred_positions_sim_all_C = \
        sim_all(start_position, sim, gui, commands, obstacle_positions)

    pred_positions_with_real_sensor_data_A, pred_positions_with_real_sensor_data_B, pred_positions_with_real_sensor_data_C = \
        sim_sens(start_position, sim, gui, commands, obstacle_positions)

    pred_positions_in_loop_A, pred_positions_in_loop_B, pred_positions_in_loop_C = \
        loop(start_position, sim, gui, commands, obstacle_positions)


    return [real_positions, pred_positions_sim_all_A, pred_positions_with_real_sensor_data_A, pred_positions_in_loop_A],\
           [real_positions, pred_positions_sim_all_B, pred_positions_with_real_sensor_data_B, pred_positions_in_loop_B],\
           [real_positions, pred_positions_sim_all_C, pred_positions_with_real_sensor_data_C, pred_positions_in_loop_C],\
           obstacle_positions

def get_distances(positions1, positions2):
    return [np.sqrt((p2 - p1)[0] ** 2 + (p2 - p1)[1] ** 2) for p1, p2 in zip(positions1, positions2)]

from tkinter import *

def transform_pos(pos, min_x, max_x, min_y, max_y, new_width, new_height):
    x = pos
    old_width = max_x - min_x #3
    old_height = max_y - min_y #2
    new_x = [x[0] - min_x, np.abs(x[1]  - max_y)]
    new_x = [new_width / old_width * new_x[0], new_height/ old_height * new_x[1]]
    return new_x

def visualize_positions_think_path(data, obstacles, to_draw):

    master = Tk()

    w = Canvas(master, width=1000, height=1000)
    w.pack()
    w.create_rectangle(0, 0, 600, 400, fill='white')
    for i in range(6):
        w.create_line(i * 100, 0, i * 100, 400, fill='grey')
    for i in range(4):
        w.create_line(0, i * 100, 600, i * 100, fill='grey')
    #draw border
    w.create_rectangle(0,0,600,400, width=5)
    #draw_obstacles
    for p in obstacles:
        p = transform_pos(p, -1.5, 1.5, 0, 2, 600, 400)
        r = 10
        w.create_oval(p[0] - r, p[1] - r,p[0] + r, p[1] + r , fill='yellow')

    #draw real positions
    for idx, pos in enumerate(data[0]):
        r = 2
        r1 = 10
        if idx == len(data[1]) - 1:
            pt = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            w.create_oval(pt[0] - r1, pt[1] - r1, pt[0] + r1, pt[1] + r1, fill='green')
            w.create_text(pt[0], pt[1], text='S')
        if idx + 1 < len(data[0]):
            pos = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            next_pos = data[0][idx + 1]
            next_pos = transform_pos(next_pos, -1.5, 1.5, 0, 2, 600, 400)
            if to_draw[0]:
                w.create_line(pos[0], pos[1], next_pos[0], next_pos[1], fill='blue', width=3)
        #w.create_oval(pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r, fill='black')

    for idx, pos in enumerate(data[1]):
        r = 2
        r1 = 10
        #if idx == len(data[1]) - 1:
        #    pt = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            #w.create_oval(pt[0] - r1, pt[1] - r1, pt[0] + r1, pt[1] + r1, fill='green')
            #w.create_text(pt[0], pt[1], text='A')
        if idx + 1 < len(data[1]):
            pos = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            next_pos = data[1][idx + 1]
            next_pos = transform_pos(next_pos, -1.5, 1.5, 0, 2, 600, 400)
            if to_draw[1]:
                w.create_line(pos[0], pos[1], next_pos[0], next_pos[1], fill='red', width=3)
        #w.create_oval(pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r, fill='black')

    for idx, pos in enumerate(data[2]):
        r = 2
        r1 = 10
        if idx + 1 < len(data[2]):
            pos = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            next_pos = data[2][idx + 1]
            next_pos = transform_pos(next_pos, -1.5, 1.5, 0, 2, 600, 400)
            if to_draw[2]:
                w.create_line(pos[0], pos[1], next_pos[0], next_pos[1], fill='orange', width=3)
        #w.create_oval(pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r, fill='black')
        if idx == len(data[2]) - 1:
            r1 = 10
            pt = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            w.create_oval(pt[0] - r1, pt[1] - r1, pt[0] + r1, pt[1] + r1, fill='green')
            w.create_text(pt[0], pt[1], text='RS')
    for idx, pos in enumerate(data[3]):
        r = 2
        r1 = 10
        if idx == len(data[3]) - 1:
            pt = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            w.create_oval(pt[0] - r1, pt[1] - r1, pt[0] + r1, pt[1] + r1, fill='green')
            w.create_text(pt[0], pt[1], text='L')
        if idx + 1 < len(data[3]):
            pos = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            next_pos = data[3][idx + 1]
            next_pos = transform_pos(next_pos, -1.5, 1.5, 0, 2, 600, 400)
            if to_draw[3]:
                w.create_line(pos[0], pos[1], next_pos[0], next_pos[1], fill='green', width=3)
        #w.create_oval(pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r, fill='black')

    mainloop()

data_A, data_B, data_C, obstacles = get_all_positions_data()

visualize_positions_think_path(data_C, obstacles, [True, False,True,True])
