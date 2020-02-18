import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

import global_config as c
from simulator import Simulator
from data import Data
from gui import GUI
from compare_agent import Agent
from stopwatch import Stopwatch
from plot import Plot
import sys

epochs = sys.argv[1]
mini_epochs = sys.argv[2]
time_steps = sys.argv[3]
learning_rate = sys.argv[4]
model_name = sys.argv[5]
v_noise = sys.argv[6]
v_drop = sys.argv[7]


import faulthandler


def train():
    faulthandler.enable()

    # CONSTANTS
    NUM_EPOCHS = int(epochs)  # Amount of epochs of size NUM_MINI_EPOCHS
    NUM_MINI_EPOCHS = int(mini_epochs)  # Amount of mini-epochs of size NUM_TIME_STEPS
    NUM_TIME_STEPS = int(time_steps)
    LEARNING_RATE = float(learning_rate)
    name = model_name

    s = Stopwatch()
    sim = Simulator(mode=1, stopwatch=s)

    agents = []
    # a1 = Agent(id='A8', color='red', init_pos=np.array([0., 1.]), lr=LEARNING_RATE , gui=gui, sim=sim, num_epochs=NUM_EPOCHS)
    # agents.append(a1)
    if v_noise == 'True':
        val_noise = float(sys.argv[8])
        if v_drop == 'True':
            val_drop = float(sys.argv[9])
            a2 = Agent(id='Z2', color='green', init_pos=np.array([-0.01, 0.8]), lr=LEARNING_RATE, gui=None, sim=sim,
                        num_epochs=NUM_EPOCHS, v_noise=val_noise, v_drop=val_drop)
        else:
            a2 = Agent(id='Z2', color='green', init_pos=np.array([-0.01, 0.8]), lr=LEARNING_RATE, gui=None, sim=sim,
                       num_epochs=NUM_EPOCHS, v_noise=val_noise)
    elif v_drop == 'True':
        val_drop = float(sys.argv[9])
        a2 = Agent(id='Z2', color='green', init_pos=np.array([-0.01, 0.8]), lr=LEARNING_RATE, gui=None, sim=sim,
                   num_epochs=NUM_EPOCHS, v_drop=val_drop)

    else :
        a2 = Agent(id='Z2', color='green', init_pos=np.array([-0.01, 0.8]), lr=LEARNING_RATE, gui=None, sim=sim,
                   num_epochs=NUM_EPOCHS)
    agents.append(a2)  # [0., 1.4]

    # obstacles = []
    # obstacles = create_obstacles(4, gui=gui, sim=sim, positi   ons=np.array([[0., 1.]]))
    obstacles = create_obstacles(1, gui=None, sim=sim)

    for a in agents:
        a.register_agents(obstacles)

    # --------------
    # MAIN PROGRAM
    # --------------

    # Fuer alle Epochen
    for epoch in range(1, NUM_EPOCHS + 1):

        # Reset all relevant variables
        from_step = 0
        to_step = 0

        # Clear gradients, simulator and data
        reset_agents(agents)
        reset_obstacles(obstacles)
        # obstacles_step(obstacles, 1)
        for mini_epoch in range(NUM_MINI_EPOCHS):

            # Define the next from and to step
            from_step = mini_epoch * NUM_TIME_STEPS
            to_step = from_step + NUM_TIME_STEPS

            # Set obstacle agents
            scenario = mini_epoch % len(c.LEARNING_SCENARIOS)
            create_obstacle_path(obstacles, c.LEARNING_SCENARIOS[scenario], NUM_TIME_STEPS + 1)

            stand_still = False
            if epoch > 100 and epoch % 2 == 1:
                stand_still = True
            else:
                stand_still = False

            # Create random motor commands, get inputs and targets for next x timesteps
            create_inputs_and_targets(agents, from_step, to_step, stand_still)

            for a in agents:
                a.learning_mini_epoch(from_step, to_step, NUM_TIME_STEPS)

        if epoch % 1 == 0 and epoch < NUM_EPOCHS:
            for a in agents:
                printAndPlot(epoch, a)

        if epoch % 20 == 0 and epoch < NUM_EPOCHS:
            for a in agents:
                a.save('./created_models/mode_' + name + '_epoch' + str(epoch) + '.pt')

        for a in agents:
            a.scheduler.step(a.losses[-1])

        # --End of epoch

    # End of all Epochs
    # Plot
    for a in agents:
        printAndPlot(epoch, a, final=True)

    # Save model
    for a in agents:
        a.save('./saves/mode_' + str(a.id) + '_final.pt')
        a.save('./compare_models/' + model_name)

    # Save mean losses
    for a in agents:
        np.savetxt("./results/agent" + str(a.id) + "_loss.csv", a.mean_losses, delimiter=";")
        np.savetxt("./results/agent" + str(a.id) + "_loss_pos.csv", a.mean_losses_positions, delimiter=";")
        np.savetxt("./results/agent" + str(a.id) + "_loss_sens.csv", a.mean_losses_sensors, delimiter=";")
        np.savetxt("./results/agent" + str(a.id) + "_loss_acc.csv", a.mean_losses_accelerations, delimiter=";")

        a.plot.save(a.id)

    s.summary()


# Create the data for each time step, for each agent
def create_inputs_and_targets(agents, from_step, to_step, stand_still=False):
    time_steps = to_step - from_step

    for t in range(time_steps):
        for a in agents:

            a.data.create_inputs(1, a, stand_still)

            if c.INPUT_SENSOR_DIM > 0:
                sensor_data = a.sim.calc_sensor_data(from_step + t, from_step + t + 1, a)
                a.data.sensors.append(sensor_data)

    # At last, calculate the velocity and position for the motor commands of the very last time step
    # since the velocity is used as target for learning.
    # E.g. with num_time_steps = 20 the loop creates velocity[0:19] and motordata[0:19]
    # The targets that function "get_targets_block" returns are velocity[1:20].
    # Thus we must caluclate velocity[20] by executing motordata[19].
    # After that, we decrease the data-indices, so that the next call of create_inputs will again start at
    # writing to sensordata[20].
    for a in agents:
        a.data.create_inputs(1, a)
        a.data.position_deltas.change_curr_idx(-1)
        a.data.positions.change_curr_idx(-1)
        a.data.motor_commands.change_curr_idx(-1)
        a.data.accelerations.change_curr_idx(-1)


def reset_agents(agents):
    for a in agents:
        a.reset_learning()


def reset_obstacles(obstacles):
    for o in obstacles:
        o.data.reset()


def __obstacles_step(obstacles, num_time_steps):
    for o in obstacles:
        positions = np.repeat(o.init_pos[np.newaxis, :], num_time_steps, axis=0)
        o.data.positions.append(positions)


def create_obstacles(number, gui, sim, positions=None):
    obstacles = []
    for obstacle_index in range(number):

        if positions is None:
            # x between -1.5 and 1.5
            x = (np.random.rand() * 3) - 1.5
            # y between 0 and 2
            y = np.random.rand() * 2

            position = np.array([x, y])

        else:
            position = positions[obstacle_index]

        o = Agent(id='O' + str(obstacle_index), color='gray', init_pos=position, lr=None, gui=gui, sim=sim,
                  is_obstacle=True)

        obstacles.append(o)

        if o.gui_att is not None:
            o.gui_att.show_target = False
            o.gui_att.show_simulated_positions = False

    return obstacles


def create_obstacle_path(obstacles, pathtype, num_time_steps):
    # Debugging
    # print("Function: create_obstacle_path")
    # print("pathtype", pathtype)
    for o in obstacles:

        positions = np.zeros([num_time_steps, c.INPUT_POSITION_DIM])

        # Get initial position
        # x between -1.5 and 1.5
        x = (np.random.rand() * 3) - 1.5
        # y between 0 and 2
        y = np.random.rand() * 2

        positions[0] = np.array([x, y])

        if pathtype is 'static':
            vel_x = 0.
            vel_y = 0.

            velocities = np.array([vel_x, vel_y])

            # Get all positions of one epoch
            for t in range(1, num_time_steps):
                positions[t] = positions[t - 1] + velocities

        elif pathtype is 'alone':
            for t in range(0, num_time_steps):
                positions[t] = np.array([1000., 1000.])

        elif 'line' in pathtype:
            vel_x = np.random.rand() * c.CLAMP_TARGET_VELOCITY_VALUE
            vel_x -= c.CLAMP_TARGET_VELOCITY_VALUE / 2.
            vel_y = np.random.rand() * c.CLAMP_TARGET_VELOCITY_VALUE
            vel_y -= c.CLAMP_TARGET_VELOCITY_VALUE / 2.

            velocities = np.array([vel_x, vel_y])

            # Get all positions of one epoch
            for t in range(1, num_time_steps):
                positions[t] = positions[t - 1] + velocities

                if 'acc' in pathtype:
                    velocities *= 1.05


        elif 'curve' in pathtype:
            # Change direction by a random fraction of it between -0.3 and 0.3
            delta_angle = np.random.rand() * (np.pi / 50.)
            radius = np.random.rand() * 0.2 + 0.1
            # delta_angle = np.pi/20
            # radius = 0.5
            # radius = np.linalg.norm(positions[0])
            angle = np.random.rand() * np.pi * 2

            center = positions[0] - np.array([np.math.cos(angle), np.math.sin(angle)]) * radius

            for t in range(1, num_time_steps):
                angle = (angle + delta_angle) % (np.pi * 2)
                next_x = center[0] + np.math.cos(angle) * radius
                next_y = center[1] + np.math.sin(angle) * radius

                positions[t] = np.array([next_x, next_y])

                if 'acc' in pathtype:
                    delta_angle *= 1.01

        elif 'designed' in pathtype:
            for t in range(0, num_time_steps):
                positions[t] = np.array([1000., 1000.])

        o.data.positions.append(positions)
        o.data.positions.change_curr_idx(-1)


def printAndPlot(epoch, a, final=False):
    plot_losses = []
    f = "{0:0.7f}"

    losses = np.asarray(a.losses)
    loss_mean = np.mean(losses)
    a.mean_losses = np.append(a.mean_losses, loss_mean)
    plot_losses.append(loss_mean)

    losses_positions = np.asarray(a.losses_positions)
    loss_positions_mean = np.mean(losses_positions)
    a.mean_losses_positions = np.append(a.mean_losses_positions, loss_positions_mean)
    plot_losses.append(loss_positions_mean)

    if c.OUTPUT_SENSOR_DIM > 0:
        losses_sensors = np.asarray(a.losses_sensors)
        loss_sensors_mean = np.mean(losses_sensors)
        a.mean_losses_sensors = np.append(a.mean_losses_sensors, loss_sensors_mean)
        plot_losses.append(loss_sensors_mean)

    if c.OUTPUT_ACCELERATION_DIM > 0:
        losses_accelerations = np.asarray(a.losses_accelerations)
        loss_accelerations_mean = np.mean(losses_accelerations)
        a.mean_losses_accelerations = np.append(a.mean_losses_accelerations, loss_accelerations_mean)
        plot_losses.append(loss_accelerations_mean)

    string = ''

    if not final:
        string += "E" + str(epoch)
    else:
        string += "END"

    string += "\t" + a.id + \
              "\tloss=" + str(f.format(loss_mean)) + \
              "\tl_pos=" + str(f.format(loss_positions_mean))

    if c.OUTPUT_SENSOR_DIM > 0:
        string += "\tl_sen=" + str(f.format(loss_sensors_mean))

    if c.OUTPUT_ACCELERATION_DIM > 0:
        string += "\tl_acc=" + str(f.format(loss_accelerations_mean))

    print(string)

    a.plot.plot(plot_losses, persist=final)


train()