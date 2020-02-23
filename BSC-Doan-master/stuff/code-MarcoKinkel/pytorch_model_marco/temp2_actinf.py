import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import global_config as c
from simulator import Simulator
from data import Data
from gui import GUI
from compare_agent import Agent
from stopwatch import Stopwatch
import datetime

import sys

import matplotlib._color_data as mcd
import matplotlib.pyplot as plt

import faulthandler

compare = False
#compare stuff
if compare:
    comp_model_path = sys.argv[1] #Model which is tested
    comp_id = sys.argv[2] #id to save
    compare_step = sys.argv[3]#one-step or two-step actinf
comp_data = []#distance from a to b over time


def actinf():
    # comp_start = comp.Start()

    faulthandler.enable()

    # CONSTANTS
    NUM_ALL_STEPS = 240  # Amount of epochs of size NUM_MINI_EPOCHS
    NUM_TIME_STEPS = 10
    TARGET_CHANGE_FREQUENCY = 40
    ACTINF_ITERATIONS = 10
    LEARNING_RATE = .01  # .01

    s = Stopwatch()
    sim = Simulator(mode=1, stopwatch=s)
    gui = GUI()
    agents = []
    obstacles = []


    if c.MODE == 10:
        a1 = Agent(
            id='A',
            color='red',
            init_pos=np.array([0., 1.]),
            gui=gui,
            sim=sim,
            modelfile='./saves/mode_T15_final.pt'  # test_act_2
        )
        a2 = Agent(
            id='B',
            color='green',
            position_loss_weight_actinf=0.0,
            sensor_loss_weight_actinf=1000.0,
            seek_proximity=True,
            show_sensor_plot=False,
            init_pos=np.array([-0.01, 0.8]),  # davon sind motorcommands2 [0, 1.25]), #m3 schoener! [-0.01, 0.8]),
            gui=gui, sim=sim,
            modelfile='./compare_models/v_0.6_0.1'
        )

        agents.append(a1)
        agents.append(a2)

        # obstacle = create_obstacles(1, gui, sim, positions=[np.array([0, 1.])], color=None, name=None)

        # a1.register_agents(a2)
        a2.register_agents([a1])

        a2.gui_att.show_target = False
        # a1.gui_att.show_predictions = False
        # a2.gui_att.show_predictions = False
        a1.gui_att.show_simulated_positions = False
        a2.gui_att.show_simulated_positions = False

        def set_targets(from_step, c):
            #no targets in this szenario
            a_pos = a1.data.positions.get(from_step)
            b_pos = a2.data.positions.get(from_step)

            # Plot the distance of B to A
            distance = np.linalg.norm(b_pos - a_pos)
            #a2.performances = np.concatenate([a2.performances, [distance]])
            #a2.plot.plot([distance])
            comp_data.append(distance)

    # s.start('gui')
    gui.draw()
    # s.stop('gui')

    for a in agents:
        # Set the position before executing time step 0
        a.data.position_deltas.change_curr_idx(1)
        a.data.velocities.change_curr_idx(1)
        a.data.accelerations.change_curr_idx(1)
        a.data.states.append_single(torch.stack(a.net.init_hidden()).data.numpy()[:, -1, :, :])
        # a.data.scv.change_curr_idx(1)

    for a in agents:
        # If the sensor inputs should be predicted, the initial sensor inputs at time step 0
        # must be given. This must happen in an extra loop after all initial positions have been calculated
        if c.OUTPUT_SENSOR_DIM > 0:
            a.data.sensors.append(a.sim.calc_sensor_data(0, 1, a))

    gui.draw()

    # --------------
    # MAIN PROGRAM
    # --------------

    # For all single time steps
    data_capture = []
    loaded_commands = torch.load('commands3.pt')
    for t in range(NUM_ALL_STEPS):

        from_step = t
        to_step = from_step + NUM_TIME_STEPS

        comp_target = int(t / TARGET_CHANGE_FREQUENCY)

        # GUI
        gui.update_time_step(from_step, to_step)

        set_targets(from_step, comp_target)

        obstacles_step(obstacles, 1)
        for o in obstacles:
            o.gui_att.update_position(o.data.positions.get(from_step))

        # Do active inference simultaneously for all agents
        for i in range(ACTINF_ITERATIONS):

            pre_iteration(agents, from_step)

            for input_t in range(NUM_TIME_STEPS):
                for a in agents:
                    step = t % TARGET_CHANGE_FREQUENCY
                    if step < (TARGET_CHANGE_FREQUENCY / 4):
                        command = [1, 1, 0, 0]
                    elif step < 2 * (TARGET_CHANGE_FREQUENCY / 4):
                        command = [1, 0, 1, 0]
                    elif step < 3 * (TARGET_CHANGE_FREQUENCY / 4):
                        command = [0, 1, 0, 1]
                    else:
                        command = [0, 0, 1, 1]
                    if a.id == 'A':
                        a.predict_future(from_step, input_t, no_sim=True, no_sim_command=np.array(command))
                    else:
                        a.predict_future(from_step, input_t)

            # Prediction into future is done, now calculate error, perform bwpass
            # and apply gradients to motor commands
            for a in agents:
                velinf_needed = (c.MODE is 10 and a.id is 'B') or c.MODE is 82
                if compare:
                    if compare_step == '1step':
                        velinf_needed = False
                # velinf_needed = True

                if velinf_needed:
                    # Perform velocity inference:
                    # Use the sensor loss to not adapt the motor commands (does not work well),
                    # but to adapt the velocities that would be needed to follow the gradient.
                    # These are then written to data.actinf_targets.
                    a.actinf(from_step, to_step, 1, velinf=True)
                    B_pos = a.data.positions.get(from_step)
                    # print('position of B', B_pos)

                    # Reset sensor-gradient-following agent after performing velocity inference
                    post_iteration([a], NUM_TIME_STEPS)
                    pre_iteration([a], from_step)

                    # Now follow the gradient calculated with actinf_targets
                    for input_t in range(NUM_TIME_STEPS):
                        a.predict_future(from_step, input_t)

                    tmp_position_loss_weight_actinf = a.position_loss_weight_actinf
                    tmp_sensor_loss_weight_actinf = a.sensor_loss_weight_actinf
                    a.position_loss_weight_actinf = 1.0
                    a.sensor_loss_weight_actinf = 0.

                # Perform motor inference
                #only for agent B, because has fix commands
                if a.id == 'B':
                    a.actinf(from_step, to_step, LEARNING_RATE)

                # If agent should follow sensor gradient, restore the weights
                if velinf_needed:
                    a.position_loss_weight_actinf = tmp_position_loss_weight_actinf
                    a.sensor_loss_weight_actinf = tmp_sensor_loss_weight_actinf

                # Now we have new motor commands for the same NUM_TIME_STEPS time steps
            # --------------------

            # At the end of an iteration, reset all indices and restore the simulator-savepoints
            post_iteration(agents, NUM_TIME_STEPS)

        if c.SHOW_SENSOR_PLOT is True:
            s.start('sensorplot')

            if c.SHOW_SENSOR_PLOT_STEP_BY_STEP is False:
                sensor_data = a.data.sensors.get(from_step)

                if c.OUTPUT_SENSOR_DIM > 0:
                    predictions = a.actinf_sensor_predictions[0]
                else:
                    predictions = np.zeros_like(sensor_data)

                a.sensorplot.update(sensor_data, predictions, "t = " + str(from_step))
                plotname = a.id + "_" + "{0:4}".format(from_step).replace(" ", "0")
                a.sensorplot.save(plotname)

            else:
                for t in range(len(a.actinf_sensor_predictions)):
                    for a in agents:
                        # Draw sensor subplot
                        if a.show_sensor_plot and c.INPUT_SENSOR_DIM > 0:
                            # sensor_data = a.data.sensors.get(from_step-1)

                            # # Performance issues: only draw if sensor data changed
                            # previous_sensor_data = a.data.sensors.get(from_step-2)
                            # if not np.array_equal(sensor_data, previous_sensor_data):
                            #     a.sensorplot.update(sensor_data, a.actinf_sensor_predictions[0])

                            sensor_data = a.data.sensors.get(from_step + t)

                            # Performance issues: only draw if sensor data changed
                            a.sensorplot.update(sensor_data, a.actinf_sensor_predictions[t],
                                                "t = " + str(from_step) + " + " + str(t + 1))
                            plotname = a.id + "_" + "{0:4}".format(from_step).replace(" ", "0") + "+" + str(t)
                            a.sensorplot.save(plotname)

            s.stop('sensorplot')

        for a in agents:
            a.data.scv.change_curr_idx(1)

        # Perform real step

        for a in agents:
            if a.id == 'A':
                com = loaded_commands[t]
                if False:
                    step = t % TARGET_CHANGE_FREQUENCY
                    speed = 0.7
                    if step < (TARGET_CHANGE_FREQUENCY / 8): #Agent nach norden
                        com = [speed, speed, 0, 0]
                    elif step < 2 * (TARGET_CHANGE_FREQUENCY / 8):
                        com = [0, 0, speed, speed]
                    elif step < 3 * (TARGET_CHANGE_FREQUENCY / 8): #Agent nach westen
                        com = [0, speed, 0, speed]
                    elif step < 4 * (TARGET_CHANGE_FREQUENCY / 8):
                        com = [speed, 0, speed, 0]
                    elif step < 5 * (TARGET_CHANGE_FREQUENCY / 8):#Agent nach sueden
                        com = [0, 0, speed, speed]
                    elif step < 6 * (TARGET_CHANGE_FREQUENCY / 8):
                        com = [speed, speed, 0, 0]
                    elif step < 7 * (TARGET_CHANGE_FREQUENCY / 8):#Agent nach osten
                        com = [speed, 0, speed, 0]
                    elif step < TARGET_CHANGE_FREQUENCY:
                        com = [0, speed, 0, speed]
                action = a.real_step(from_step, need_action=False, command=com)
                data_capture.append(action)
            else:
                a.real_step(from_step)

            # Calculate distance to target
            a.target_steps_total += 1
            my_pos = a.data.positions.get_relative(0)
            targets_abs_position = a.data.get_actinf_targets_block(from_step, from_step + 1)[0]

            distance = np.linalg.norm(my_pos - targets_abs_position)

            # If distance is smaller than a threshold, increment counter
            if distance < 0.03:
                a.on_target_steps += 1
            else:
                a.on_target_steps = 0

        # if c.TAKE_SCREENSHOT is True:
        # gui.save_screenshot("{0:4}".format(from_step).replace(" ", "0"))

        # agents[0].sensorplot.save("{0:4}".format(from_step).replace(" ", "0"))
    # End

    for a in agents:
        filename = "chasing_smaller" + get_stage() + "_agent" + str(a.id) + "_mode" + str(c.MODE)
        # a.plot.save(filename)
        np.savetxt("./results/" + filename + ".csv", np.asarray(a.performances), delimiter=";")

    s.summary()
    # np.save('motorcommands', data_capture)
    #torch.save(data_capture, 'commands3.pt')


def pre_iteration(agents, from_step):
    for a in agents:
        a.actinf_position_predictions = []
        a.actinf_sensor_predictions = []
        a.actinf_inputs = []
        a.actinf_previous_state = a.data.states.get(from_step)


def post_iteration(agents, num_time_steps):
    for a in agents:
        a.data.positions.change_curr_idx(-1 * num_time_steps)
        a.data.position_deltas.change_curr_idx(-1 * num_time_steps)
        a.data.velocities.change_curr_idx(-1 * num_time_steps)
        a.data.accelerations.change_curr_idx(-1 * num_time_steps)
        a.data.scv.change_curr_idx(-1 * num_time_steps)


def get_stage():
    if c.OUTPUT_SENSOR_DIM > 0:
        return '3'
    elif c.INPUT_SENSOR_DIM > 0:
        return '2'
    else:
        return '1'


def obstacles_step(obstacles, num_time_steps):
    for o in obstacles:
        if o.path_scenario is None:
            positions = np.repeat(o.init_pos[np.newaxis, :], num_time_steps, axis=0)
            o.data.positions.append(positions)

        else:
            for t in range(num_time_steps):

                if o.data.positions.curr_idx == 0:
                    current_pos = o.init_pos
                else:
                    current_pos = o.data.positions.get_relative(-1)

                if 'static' in o.path_scenario:
                    next_position = current_pos

                elif 'linear' in o.path_scenario:
                    next_position = current_pos + o.path_velocities

                    if 'acc' in o.path_scenario:
                        o.velocities *= 1.05

                elif 'circle' in o.path_scenario:
                    angle = (o.path_angle + o.path_delta_angle) % (np.pi * 2)
                    next_x = o.path_center[0] + np.math.cos(angle) * o.path_radius
                    next_y = o.path_center[1] + np.math.sin(angle) * o.path_radius

                    next_position = np.array([next_x, next_y])

                    if 'acc' in o.path_scenario:
                        o.path_delta_angle *= 1.05

                o.data.positions.append([next_position])


def create_obstacles(number, gui, sim, positions=None, color=None, name=None):
    if color is None:
        color = 'gray'

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

        if name is None:
            name = 'O' + str(obstacle_index)

        o = Agent(id=name, color=color, init_pos=position, lr=None, gui=gui, sim=sim, is_obstacle=True)

        obstacles.append(o)
        o.gui_att.show_target = False

    return obstacles
if compare:
    if compare_step == '1step':
        actinf()
        np.save('./datas/' + str(comp_id) + '-1', comp_data)
    if compare_step == '2step':
        actinf()
        np.save('./datas2/' + str(comp_id) + '-2', comp_data)
else:
    actinf()