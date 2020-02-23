import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import global_config as c
from compare_simulator import Simulator
from data import Data
from gui import GUI
from compare_agent import Agent
from stopwatch import Stopwatch
import datetime

import sys

import matplotlib._color_data as mcd
import matplotlib.pyplot as plt


import faulthandler

def actinf():

    #comp_start = comp.Start()

    faulthandler.enable()

    fix_targets = False
    one_step_actinf = False

    # CONSTANTS
    NUM_ALL_STEPS = 240  # Amount of epochs of size NUM_MINI_EPOCHS
    NUM_TIME_STEPS = 10
    TARGET_CHANGE_FREQUENCY = 30
    ACTINF_ITERATIONS = 10
    LEARNING_RATE = .01         # .01

    s = Stopwatch()
    sim = Simulator(mode=1, stopwatch=s)
    gui = GUI()
    agents = []
    obstacles = []


    if c.MODE == -1:
        a1 = Agent(id='A', color='red', init_pos=np.array([0., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')

        agents.append(a1)

        a1.gui_att.show_simulated_positions = True

        def set_targets(from_step):

            # All agents follow the same target
            if (from_step % TARGET_CHANGE_FREQUENCY == 0):
                target = a1.create_target(TARGET_CHANGE_FREQUENCY, NUM_TIME_STEPS)


    if c.MODE == 0:
        a1 = Agent(id='A', color='red', init_pos=np.array([0., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a2 = Agent(id='B', color='green', init_pos=np.array([-1., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')

        agents.append(a1)
        agents.append(a2)

        a1.register_agents(a2)
        a2.register_agents(a1)

        a2.gui_att.show_target = False
        # a1.gui_att.show_predictions = False
        # a2.gui_att.show_predictions = False
        a1.gui_att.show_simulated_positions = False
        a2.gui_att.show_simulated_positions = False

        def set_targets(from_step):
            # All agents follow the same target
            if (from_step % TARGET_CHANGE_FREQUENCY == 0):
                # Create new target for the next target period plus the future prediction time steps
                target = a1.create_target(TARGET_CHANGE_FREQUENCY, NUM_TIME_STEPS)
                a2.create_target(TARGET_CHANGE_FREQUENCY, NUM_TIME_STEPS, position=target)

    if c.MODE == 1:
        a1 = Agent(id='A', color='red', init_pos=np.array([0., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a2 = Agent(id='B', color='green', init_pos=np.array([-1., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')

        agents.append(a1)
        agents.append(a2)

        a1.register_agents(a2)
        a2.register_agents(a1)

        a2.gui_att.show_target = False
        # a1.gui_att.show_predictions = False
        # a2.gui_att.show_predictions = False
        a1.gui_att.show_simulated_positions = False
        a2.gui_att.show_simulated_positions = False

        def set_targets(from_step):
            # Target of agent 1 is random
            # Target of agent 2 is position of agent 1

            # Change target of agent 1
            if (from_step % TARGET_CHANGE_FREQUENCY == 0):
                # Create new target for the next target period plus the future prediction time steps
                a1.create_target(TARGET_CHANGE_FREQUENCY, NUM_TIME_STEPS)

            # Every time step
            a_pos = a1.data.positions.get(from_step)
            a2.set_target(a_pos, 1, NUM_TIME_STEPS)

            # Plot the distance of B to A
            b_pos = a2.data.positions.get(from_step)
            distance = np.linalg.norm(b_pos - a_pos)

            a2.performances = np.concatenate([a2.performances, [distance]])
            a2.plot.plot([distance])

            # a1.plot.plot([target_steps_total], init_distance)

    if c.MODE == 2:
        a1 = Agent(id='A', color='red', init_pos=np.array([0., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a2 = Agent(id='B', color='green', init_pos=np.array([-1., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')

        agents.append(a1)
        agents.append(a2)

        a1.register_agent(a2)
        a2.register_agent(a1)

        a2.gui_att.show_target = False
        # a1.gui_att.show_predictions = False
        # a2.gui_att.show_predictions = False
        a1.gui_att.show_simulated_positions = False
        a2.gui_att.show_simulated_positions = False

        def set_targets(from_step):
            # Target of agent 1 is moving on a line
            # Target of agent 2 is position of agent 1
            if (from_step % TARGET_CHANGE_FREQUENCY == 0):
                a1.create_target_line(TARGET_CHANGE_FREQUENCY, NUM_TIME_STEPS)

            a1.gui_att.update_target(a1.data.actinf_targets.get(from_step))
            a2.set_target(1, NUM_TIME_STEPS, position=a1.data.positions.get(from_step))

    if c.MODE == 3:
        pos_a1 = np.array([-1., 1.])
        pos_obstacle = np.array([-0.4, 1.0])
        pos_target = np.array([1., 1.])

        a1 = Agent(id='A', color='red', init_pos=pos_a1, gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt', stopwatch=s)

        obstacles = create_obstacles(1, gui, sim, [pos_obstacle])
        # Initial step for obstacles, so that they are always 1 step further of agents
        obstacles_step(obstacles, NUM_TIME_STEPS)

        agents.append(a1)

        a1.register_agents(obstacles)

        # a1.gui_att.show_predictions = False
        # a2.gui_att.show_predictions = False
        a1.gui_att.show_simulated_positions = False

        def set_targets(from_step):
            # a2 stands still in the middle
            # a1 on collision course of a2
            if (from_step % TARGET_CHANGE_FREQUENCY == 0):
                current_target = a1.data.actinf_targets.get(from_step)
                if all(np.equal(current_target, np.zeros(2))):
                    a1.create_target(TARGET_CHANGE_FREQUENCY, NUM_TIME_STEPS, np.array([1., 1.]))
                else:
                    current_target[0] *= -1
                    a1.create_target(TARGET_CHANGE_FREQUENCY, NUM_TIME_STEPS, current_target)

    if c.MODE == 4:
        a1 = Agent(id='A', color='red', position_loss_weight=0., init_pos=np.array([0., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a2 = Agent(id='B', color='green', init_pos=np.array([-1., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')

        agents.append(a1)
        agents.append(a2)

        a1.register_agents(a2)
        a2.register_agents(a1)

        a1.gui_att.show_target = False
        a2.gui_att.show_target = False
        # a1.gui_att.show_predictions = False
        # a2.gui_att.show_predictions = False
        a1.gui_att.show_simulated_positions = False
        a2.gui_att.show_simulated_positions = False

        def set_targets(from_step):
            # Target of agent 1 is random
            # Target of agent 2 is position of agent 1

            # Change target of agent 1
            if (from_step % TARGET_CHANGE_FREQUENCY == 0):
                # Create new target for the next target period plus the future prediction time steps
                a1.create_target(TARGET_CHANGE_FREQUENCY, NUM_TIME_STEPS)

            # Every time step
            a2.set_target(1, NUM_TIME_STEPS, position=a1.data.positions.get(from_step))

    if c.MODE == 5:
        a1 = Agent(id='A', color='red', position_loss_weight=0., seek_proximity=False, init_pos=np.array([-1.4, 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a2 = Agent(id='B', color='green', position_loss_weight=0., seek_proximity=True, init_pos=np.array([-1., 1.]), gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')

        agents.append(a1)
        agents.append(a2)

        a1.register_agents(a2)
        a2.register_agents(a1)

        a1.gui_att.show_target = False
        a2.gui_att.show_target = False
        # a1.gui_att.show_predictions = False
        # a2.gui_att.show_predictions = False
        a1.gui_att.show_simulated_positions = False
        a2.gui_att.show_simulated_positions = False

        def set_targets(from_step):
            # Don't do anything
            return

    if c.MODE == 6:
        init_pos = np.array([0., 1.])
        a1 = Agent(id='A', color='red', init_pos=init_pos, gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt', stopwatch=s)
        agents.append(a1)
        a1.gui_att.show_simulated_positions = True

        def set_targets(from_step):
            # All agents follow the same target
            # if (from_step % TARGET_CHANGE_FREQUENCY == 0):
            if(from_step == 0):
                # Create new target for the next target period plus the future prediction time steps
                target = a1.create_target(1, NUM_TIME_STEPS)
                a1.init_target_distance = np.linalg.norm(init_pos - target)

            else:
                if(a1.on_target_steps < 10):
                    # Keep same target
                    target = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    target = a1.create_target(1, NUM_TIME_STEPS, position=target)

                else:
                    # Calculate the initial distance of the new target to the agent
                    my_pos = a1.data.positions.get_relative(0)

                    # Calculate the performance: how many steps did it take to get to the target
                    # normalized by the initial distance
                    # performance = a1.init_target_distance / a1.target_steps_total
                    # a1.performances = np.concatenate([a1.performances, [performance]])
                    # a.plot.plot([performance])
                    # a1.plot.plot([a1.target_steps_total], a1.init_target_distance)

                    # Finally create new target for the next target period plus the future prediction time steps
                    target = a1.create_target(1, NUM_TIME_STEPS)
                    a1.init_target_distance = np.linalg.norm(my_pos - target)
                    a1.on_target_steps = 0
                    a1.target_steps_total = 0


                my_pos = a1.data.positions.get_relative(0)
                target = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]
                dist = np.linalg.norm(my_pos - target)
                a1.performances2D.append([dist, a1.init_target_distance])
                a1.plot.plot([dist / a1.init_target_distance])

    if c.MODE == 71:
        angle = np.math.atan(1/1.5)
        dist = 0.8
        cos = np.math.cos(angle)
        sin = np.math.sin(angle)
        agent_positions = np.array(
            [
                [dist, 1],
                [dist*cos, 1 + dist*sin],
                [0, 1+dist],
                [-dist*cos, 1+dist*sin],
                [-dist,1],
                [-dist*cos, 1-dist*sin],
                [0,1-dist],
                [dist*cos, 1-dist*sin]
            ])
        target_positions = np.concatenate([agent_positions[4:], agent_positions[:4]])

        scenario = 4   # from 0 to 7

        obstacle_pos = np.array([0., 1.])

        a1 = Agent(id='A', color='red', init_pos=agent_positions[scenario], show_sensor_plot=False, gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a1.gui_att.show_simulated_positions = False

        obstacles = create_obstacles(1, gui, sim, [obstacle_pos], color='lightgreen', name='B')
        # Initial step for obstacles, so that they are always 1 step further of agents
        obstacles_step(obstacles, NUM_TIME_STEPS)

        agents.append(a1)
        a1.register_agents(obstacles)

        def set_targets(from_step):
            nonlocal scenario

            # All agents follow the same target
            # if (from_step % TARGET_CHANGE_FREQUENCY == 0):
            if(from_step == 0):
                # Create new target for the next target period plus the future prediction time steps
                a1.create_target(1, NUM_TIME_STEPS, position=target_positions[scenario])

            else:
                if(a1.on_target_steps < 1 and a1.target_steps_total < TARGET_CHANGE_FREQUENCY):
                    # Keep same target
                    target = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    a1.create_target(1, NUM_TIME_STEPS, position=target)

                else:
                    # Create new target for the next target period plus the future prediction time steps
                    scenario = (scenario + 1) % len(agent_positions)

                    if scenario == 4:
                        filename = "actinf_stage" + get_stage() + "_agent" + str(a1.id) + "_mode" + str(c.MODE)
                        a1.plot.save(filename)
                        np.savetxt("./results/"+filename+".csv", np.asarray(a1.performances2D)[:, -1, :], delimiter=";")
                        sys.exit(0)

                    new_pos = agent_positions[scenario]
                    a1.data.reset(from_step)
                    a1.data.positions.write(np.array([new_pos]), from_step)
                    a1.initial_state = torch.stack(a1.net.init_hidden()).data.numpy()[:, -1, :, :]
                    a1.actinf_position_predictions = []
                    a1.actinf_sensor_predictions = []
                    a1.actinf_inputs = []

                    a1.create_target(1, NUM_TIME_STEPS, position=target_positions[scenario])

                    # Calculate the initial distance of the new target to the agent
                    my_pos = a1.data.positions.get_relative(0)
                    init_distance = np.linalg.norm(agent_positions[scenario] - target_positions[scenario])
                    print('Initial distance to target =',str(init_distance))

                    a1.on_target_steps = 0
                    a1.target_steps_total = 0

            # Plot performance
            A_pos = a1.data.positions.get_relative(0)
            A_target_pos = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]

            # Don't plot if the position of A is not set yet -> so if it's all zero
            if not np.any(A_pos):
                return

            # Distance of A to target
            target_dist = np.linalg.norm(A_pos - A_target_pos)

            # Distance of A to B
            agent_dist = np.linalg.norm(A_pos - obstacle_pos)
            real_agent_dist = agent_dist - (2 * a1.radius)

            a1.plot.plot([target_dist, real_agent_dist])
            a1.performances2D.append([[target_dist, real_agent_dist]])

    if c.MODE == 72:
        angle = np.math.atan(1/1.5)
        dist = 0.8
        cos = np.math.cos(angle)
        sin = np.math.sin(angle)
        agent_positions = np.array(
            [
                [dist, 1],
                [dist*cos, 1 + dist*sin],
                [0, 1+dist],
                [-dist*cos, 1+dist*sin],
                [-dist,1],
                [-dist*cos, 1-dist*sin],
                [0,1-dist],
                [dist*cos, 1-dist*sin]
            ])
        target_positions = np.concatenate([agent_positions[4:], agent_positions[:4]])

        a_scenario = 4   # from 0 to 7
        b_scenario = (a_scenario + 2) % len(agent_positions)

        a1 = Agent(id='A', color='red', init_pos=agent_positions[a_scenario], gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a1.gui_att.show_simulated_positions = False

        a2 = Agent(id='B', color='lightgreen', init_pos=agent_positions[b_scenario], gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a2.gui_att.show_simulated_positions = False
        a2.gui_att.show_target = False
        a2.gui_att.show_scv_targets = False

        agents.append(a1)
        agents.append(a2)

        # B does not sense A, but A senses B

        a1.register_agents(a2)


        def set_targets(from_step):
            nonlocal a_scenario
            nonlocal b_scenario

            if(from_step == 0):
                # Create new target for the next target period plus the future prediction time steps
                a1.create_target(1, NUM_TIME_STEPS, position=target_positions[a_scenario])
                a2.create_target(1, NUM_TIME_STEPS, position=target_positions[b_scenario])

            else:
                if(a1.on_target_steps < 1 and a1.target_steps_total < TARGET_CHANGE_FREQUENCY):
                    # Keep same target
                    a_target = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    a1.create_target(1, NUM_TIME_STEPS, position=a_target)
                    b_target = a2.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    a2.create_target(1, NUM_TIME_STEPS, position=b_target)

                else:
                    # Create new target for the next target period plus the future prediction time steps
                    a_scenario = (a_scenario + 1) % len(agent_positions)
                    b_scenario = (a_scenario + 2) % len(agent_positions)

                    if a_scenario == 4:
                        filename = "actinf_stage" + get_stage() + "_agent" + str(a1.id) + "_mode" + str(c.MODE)
                        a1.plot.save(filename)
                        np.savetxt("./results/"+filename+".csv", np.asarray(a1.performances2D)[:, -1, :], delimiter=";")
                        sys.exit(0)

                    a_new_pos = agent_positions[a_scenario]
                    a1.data.reset(from_step)
                    a1.data.positions.write(np.array([a_new_pos]), from_step)
                    a1.initial_state = torch.stack(a1.net.init_hidden()).data.numpy()[:, -1, :, :]
                    a1.actinf_position_predictions = []
                    a1.actinf_sensor_predictions = []
                    a1.actinf_inputs = []

                    b_new_pos = agent_positions[b_scenario]
                    a2.data.reset(from_step)
                    a2.data.positions.write(np.array([b_new_pos]), from_step)
                    a2.initial_state = torch.stack(a2.net.init_hidden()).data.numpy()[:, -1, :, :]
                    a2.actinf_position_predictions = []
                    a2.actinf_sensor_predictions = []
                    a2.actinf_inputs = []

                    a1.create_target(1, NUM_TIME_STEPS, position=target_positions[a_scenario])
                    a2.create_target(1, NUM_TIME_STEPS, position=target_positions[b_scenario])

                    # Calculate the initial distance of the new target to the agent
                    my_pos = a1.data.positions.get_relative(0)
                    init_distance = np.linalg.norm(my_pos - target_positions[a_scenario])
                    print('Initial distance to target =',str(init_distance))

                    a1.on_target_steps = 0
                    a1.target_steps_total = 0

            # Plot performance
            A_pos = a1.data.positions.get_relative(0)
            B_pos = a2.data.positions.get_relative(0)
            A_target_pos = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]

            # Don't plot if the position of A is not set yet -> so if it's all zero
            if not np.any(A_pos):
                return

            # Distance of A to target
            target_dist = np.linalg.norm(A_pos - A_target_pos)

            # Distance of A to B
            agent_dist = np.linalg.norm(A_pos - B_pos)
            real_agent_dist = agent_dist - (2 * a1.radius)

            a1.plot.plot([target_dist, real_agent_dist])
            a1.performances2D.append([[target_dist, real_agent_dist]])

    if c.MODE == 73:
        angle = np.math.atan(1/1.5)
        dist = 0.8
        cos = np.math.cos(angle)
        sin = np.math.sin(angle)
        agent_positions = np.array(
            [
                [dist, 1],
                [dist*cos, 1 + dist*sin],
                [0, 1+dist],
                [-dist*cos, 1+dist*sin],
                [-dist,1],
                [-dist*cos, 1-dist*sin],
                [0,1-dist],
                [dist*cos, 1-dist*sin]
            ])
        target_positions = np.concatenate([agent_positions[4:], agent_positions[:4]])

        a_scenario = 4   # from 0 to 7
        b_scenario = (a_scenario + 4) % len(agent_positions)

        a1 = Agent(id='A', color='red', init_pos=agent_positions[a_scenario], gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a1.gui_att.show_simulated_positions = False

        a2 = Agent(id='B', color='lightgreen', init_pos=agent_positions[b_scenario], clamp_target_velocity_value=0.005, gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a2.gui_att.show_simulated_positions = False
        a2.gui_att.show_target = False
        a2.gui_att.show_scv_targets = False

        agents.append(a1)
        agents.append(a2)

        # B does not sense A, but A senses B

        a1.register_agents(a2)


        def set_targets(from_step):
            nonlocal a_scenario
            nonlocal b_scenario

            if(from_step == 0):
                # Create new target for the next target period plus the future prediction time steps
                a1.create_target(1, NUM_TIME_STEPS, position=target_positions[a_scenario])
                a2.create_target(1, NUM_TIME_STEPS, position=target_positions[b_scenario])

            else:
                if(a1.on_target_steps < 1 and a1.target_steps_total < TARGET_CHANGE_FREQUENCY):
                    # Keep same target
                    a_target = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    a1.create_target(1, NUM_TIME_STEPS, position=a_target)
                    b_target = a2.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    a2.create_target(1, NUM_TIME_STEPS, position=b_target)

                else:
                    # Create new target for the next target period plus the future prediction time steps
                    a_scenario = (a_scenario + 1) % len(agent_positions)
                    b_scenario = (a_scenario + 4) % len(agent_positions)

                    if a_scenario == 4:
                        filename = "actinf_stage" + get_stage() + "_agent" + str(a1.id) + "_mode" + str(c.MODE)
                        a1.plot.save(filename)
                        np.savetxt("./results/"+filename+".csv", np.asarray(a1.performances2D)[:, -1, :], delimiter=";")
                        sys.exit(0)

                    a_new_pos = agent_positions[a_scenario]
                    a1.data.reset(from_step)
                    a1.data.positions.write(np.array([a_new_pos]), from_step)
                    a1.initial_state = torch.stack(a1.net.init_hidden()).data.numpy()[:, -1, :, :]
                    a1.actinf_position_predictions = []
                    a1.actinf_sensor_predictions = []
                    a1.actinf_inputs = []

                    b_new_pos = agent_positions[b_scenario]
                    a2.data.reset(from_step)
                    a2.data.positions.write(np.array([b_new_pos]), from_step)
                    a2.initial_state = torch.stack(a2.net.init_hidden()).data.numpy()[:, -1, :, :]
                    a2.actinf_position_predictions = []
                    a2.actinf_sensor_predictions = []
                    a2.actinf_inputs = []

                    a1.create_target(1, NUM_TIME_STEPS, position=target_positions[a_scenario])
                    a2.create_target(1, NUM_TIME_STEPS, position=target_positions[b_scenario])

                    # Calculate the initial distance of the new target to the agent
                    my_pos = a1.data.positions.get_relative(0)
                    init_distance = np.linalg.norm(my_pos - target_positions[a_scenario])
                    print('Initial distance to target =',str(init_distance))

                    a1.on_target_steps = 0
                    a1.target_steps_total = 0

            # Plot performance
            A_pos = a1.data.positions.get_relative(0)
            B_pos = a2.data.positions.get_relative(0)
            A_target_pos = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]

            # Don't plot if the position of A is not set yet -> so if it's all zero
            if not np.any(A_pos):
                return

            # Distance of A to target
            target_dist = np.linalg.norm(A_pos - A_target_pos)

            # Distance of A to B
            agent_dist = np.linalg.norm(A_pos - B_pos)
            real_agent_dist = agent_dist - (2 * a1.radius)

            a1.plot.plot([target_dist, real_agent_dist])
            a1.performances2D.append([[target_dist, real_agent_dist]])

    if c.MODE == 74:
        angle = np.math.atan(1/1.5)
        dist = 0.8
        cos = np.math.cos(angle)
        sin = np.math.sin(angle)
        agent_positions = np.array(
            [
                [dist, 1],
                [dist*cos, 1 + dist*sin],
                [0, 1+dist],
                [-dist*cos, 1+dist*sin],
                [-dist,1],
                [-dist*cos, 1-dist*sin],
                [0,1-dist],
                [dist*cos, 1-dist*sin]
            ])
        target_positions = np.concatenate([agent_positions[4:], agent_positions[:4]])

        a_scenario = 4  # from 0 to 7

        a1 = Agent(id='A', color='red', init_pos=agent_positions[a_scenario], gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a1.gui_att.show_simulated_positions = False

        b_pos = (agent_positions[a_scenario] + np.array([0., 1.])) / 2.

        a2 = Agent(id='B', color='lightgreen', init_pos=b_pos, clamp_target_velocity_value=0.005, gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt')
        a2.gui_att.show_simulated_positions = False
        a2.gui_att.show_target = False
        a2.gui_att.show_scv_targets = False
        # a2.gui_att.show_predictions = False

        agents.append(a1)
        agents.append(a2)

        # B does not sense A, but A senses B

        a1.register_agents(a2)


        def set_targets(from_step):
            nonlocal a_scenario

            if(from_step == 0):
                # Create new target for the next target period plus the future prediction time steps
                a1.create_target(1, NUM_TIME_STEPS, position=target_positions[a_scenario])

                # B has same target as A
                b_init_pos = (agent_positions[a_scenario] + np.array([0., 1.])) / 2.
                b_target = target_positions[a_scenario] + b_init_pos - agent_positions[a_scenario]
                a2.create_target(1, NUM_TIME_STEPS, position=b_target)

            else:
                if(a1.on_target_steps < 1 and a1.target_steps_total < TARGET_CHANGE_FREQUENCY):
                    # Keep same target
                    a_target = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    b_target = a2.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    a1.create_target(1, NUM_TIME_STEPS, position=a_target)
                    a2.create_target(1, NUM_TIME_STEPS, position=b_target)

                else:
                    # Create new target for the next target period plus the future prediction time steps
                    a_scenario = (a_scenario + 1) % len(agent_positions)

                    if a_scenario == 4:
                        filename = "actinf_stage" + get_stage() + "_agent" + str(a1.id) + "_mode" + str(c.MODE)
                        a1.plot.save(filename)
                        np.savetxt("./results/"+filename+".csv", np.asarray(a1.performances2D)[:, -1, :], delimiter=";")
                        sys.exit(0)

                    a_new_pos = agent_positions[a_scenario]
                    a1.data.reset(from_step)
                    a1.data.positions.write(np.array([a_new_pos]), from_step)
                    a1.initial_state = torch.stack(a1.net.init_hidden()).data.numpy()[:, -1, :, :]
                    a1.actinf_position_predictions = []
                    a1.actinf_sensor_predictions = []
                    a1.actinf_inputs = []

                    b_new_pos = (a_new_pos + np.array([0., 1.])) / 2.
                    a2.data.reset(from_step)
                    a2.data.positions.write(np.array([b_new_pos]), from_step)
                    a2.initial_state = torch.stack(a2.net.init_hidden()).data.numpy()[:, -1, :, :]
                    a2.actinf_position_predictions = []
                    a2.actinf_sensor_predictions = []
                    a2.actinf_inputs = []

                    a1.create_target(1, NUM_TIME_STEPS, position=target_positions[a_scenario])

                    b_target = target_positions[a_scenario] + b_new_pos - agent_positions[a_scenario]
                    a2.create_target(1, NUM_TIME_STEPS, position=b_target)

                    # Calculate the initial distance of the new target to the agent
                    my_pos = a1.data.positions.get_relative(0)
                    init_distance = np.linalg.norm(my_pos - target_positions[a_scenario])
                    print('Initial distance to target =',str(init_distance))

                    a1.on_target_steps = 0
                    a1.target_steps_total = 0

            # Plot performance
            A_pos = a1.data.positions.get_relative(0)
            B_pos = a2.data.positions.get_relative(0)
            A_target_pos = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]

            # Don't plot if the position of A is not set yet -> so if it's all zero
            if not np.any(A_pos):
                return

            # Distance of A to target
            target_dist = np.linalg.norm(A_pos - A_target_pos)

            # Distance of A to B
            agent_dist = np.linalg.norm(A_pos - B_pos)
            real_agent_dist = agent_dist - (2 * a1.radius)

            a1.plot.plot([target_dist, real_agent_dist])
            a1.performances2D.append([[target_dist, real_agent_dist]])


    if c.MODE == 8:
        for i in range(24):
            x = (np.random.rand() * 3) - 1.5
            y = np.random.rand() * 2

            color = list(mcd.CSS4_COLORS.values())[i]
            a = Agent(id='A'+str(i), color=color, init_pos=np.array([x,y]), is_obstacle=True, gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt', stopwatch=s)
            a.register_agents(agents)
            agents.append(a)

            if(i <= 5):
                y = 0.25
                x = -1.375 + 0.5 * i
            elif(i <= 11):
                y = 0.75
                x = -1.25 + 0.5 * (i % 6)
            elif(i <= 17):
                y = 1.25
                x = -1.375 + 0.5 * (i % 6)
            else:
                y = 1.75
                x = -1.25 + 0.5 * (i % 6)

            target_pos = np.array([x, y])

            a.create_target(c.RINGBUFFER_SIZE - (NUM_TIME_STEPS+1), NUM_TIME_STEPS, target_pos)

        def set_targets(from_step):
            # Do nothing
            return

    if c.MODE == 81:
        for i in range(12):
            x = (np.random.rand() * 3) - 1.5
            y = np.random.rand() * 2

            color = list(mcd.CSS4_COLORS.values())[i]
            a = Agent(id='A'+str(i), color=color, init_pos=np.array([x,y]), is_obstacle=True, seek_proximity=False, show_sensor_plot=False, gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt', stopwatch=s)
            a.create_target(1, NUM_TIME_STEPS, position=np.array([x,y]))
            a.gui_att.show_scv_targets = True
            a.gui_att.show_target = True
            a.register_agents(agents)
            for other_a in agents:
                other_a.register_agents(a)

            agents.append(a)

        def set_targets(from_step):
            # The target of each agent is its current position
            for a in agents:
                a_pos = a.data.positions.get(from_step)
                a.create_target(1, NUM_TIME_STEPS, position=a_pos)

            # Save the distance of each agent to the closest other agent
            for a in agents:
                min_dist = 100.
                for o in a.other_agents:
                    a_pos = a.data.positions.get_relative(0)
                    o_pos = o.data.positions.get_relative(0)
                    dist = np.linalg.norm(a_pos - o_pos)
                    if dist < min_dist:
                        min_dist = dist

                # Now we have the distance to the closest agent
                a.performances.append([min_dist])

    if c.MODE == 9:
        init_pos = np.array([0., 1.])
        a1 = Agent(id='A', color='red', init_pos=init_pos, gui=gui, sim=sim, modelfile='./saves/mode_A_final.pt', stopwatch=s)
        agents.append(a1)
        a1.gui_att.show_simulated_positions = True

        obstacles = create_obstacles(20, gui, sim)
        # Initial step for obstacles, so that they are always 1 step further of agents
        a1.register_agents(obstacles)

        # scenario = target_number % len(c.LEARNING_SCENARIOS)
        scenario = 'linear'
        num_time_steps = 50

        for o in obstacles:
            o.path_scenario = scenario

            if 'linear' in scenario:
                vel_x = np.random.rand() * (3./num_time_steps)
                vel_x -= (3./num_time_steps) / 2.
                vel_y = np.random.rand() * (2./num_time_steps)
                vel_y -= (2./num_time_steps) / 2.

                o.path_velocities = np.array([vel_x, vel_y])

            elif 'circle' in scenario:
                o.path_delta_angle = np.random.rand() * (np.pi / 20.) - (np.pi / 40.)
                o.path_radius = np.random.rand() * 0.5 + 0.1
                o.path_angle = np.random.rand() * np.pi * 2
                o.path_center = o.init_pos - np.array([np.math.cos(angle), np.math.sin(angle)]) * o.path_radius

        obstacles_step(obstacles, NUM_TIME_STEPS)

        def set_targets(from_step):
            nonlocal obstacles

            # if (from_step % TARGET_CHANGE_FREQUENCY == 0):
            if(from_step == 0):
                # Create new target for the next target period plus the future prediction time steps
                target = a1.create_target(1, NUM_TIME_STEPS)
                a1.init_target_distance = np.linalg.norm(init_pos - target)

            else:
                if(a1.on_target_steps < 10):
                    # Keep same target
                    target = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    target = a1.create_target(1, NUM_TIME_STEPS, position=target)

                else:
                    # Calculate the initial distance of the new target to the agent
                    my_pos = a1.data.positions.get_relative(0)

                    # Clear old obstacles and create new ones.
                    for o in obstacles:
                        o.gui_att.update_position(np.array([100., 100.]))

                    obstacles = create_obstacles(20, gui, sim)
                    a1.other_agents = []
                    a1.register_agents(obstacles)

                    # scenario = target_number % len(c.LEARNING_SCENARIOS)
                    scenario = 'linear'
                    num_time_steps = 50

                    for o in obstacles:
                        o.path_scenario = scenario

                        if 'linear' in scenario:
                            vel_x = np.random.rand() * (3./num_time_steps)
                            vel_x -= (3./num_time_steps) / 2.
                            vel_y = np.random.rand() * (2./num_time_steps)
                            vel_y -= (2./num_time_steps) / 2.

                            o.path_velocities = np.array([vel_x, vel_y])

                        elif 'circle' in scenario:
                            o.path_delta_angle = np.random.rand() * (np.pi / 20.) - (np.pi / 40.)
                            o.path_radius = np.random.rand() * 0.5 + 0.1
                            o.path_angle = np.random.rand() * np.pi * 2
                            o.path_center = o.init_pos - np.array([np.math.cos(angle), np.math.sin(angle)]) * o.path_radius

                        # Change index so that it fits to index of agent
                        o.data.positions.change_curr_idx(from_step-1)
                        o.data.positions.append_single(o.init_pos)

                    obstacles_step(obstacles, NUM_TIME_STEPS)


                    # Finally create new target for the next target period plus the future prediction time steps
                    target = a1.create_target(1, NUM_TIME_STEPS)
                    a1.init_target_distance = np.linalg.norm(my_pos - target)
                    a1.on_target_steps = 0
                    a1.target_steps_total = 0

                my_pos = a1.data.positions.get_relative(0)
                target = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]
                dist_to_target = np.linalg.norm(my_pos - target)

                dist_to_closest_obstacle = 100.
                for o in obstacles:
                    o_pos = o.data.positions.get(from_step)
                    dist_tmp = np.linalg.norm(my_pos - o_pos)
                    dist_to_closest_obstacle = min(dist_to_closest_obstacle, dist_tmp)

                # Remove radios
                dist_to_closest_obstacle -= 0.12

                a1.performances2D.append([dist_to_target / a1.init_target_distance, dist_to_closest_obstacle])
                a1.plot.plot([dist_to_target / a1.init_target_distance, dist_to_closest_obstacle])

    if c.MODE == 10:
        a1 = Agent(
            id='A',
            color='red',
            init_pos=np.array([0., 1.]),
            gui=gui,
            sim=sim,
            modelfile='./saves/mode_T15_final.pt' #test_act_2
        )
        a2 = Agent(
            id='B',
            color='green',
            position_loss_weight_actinf=0.0,
            sensor_loss_weight_actinf=1000.0,
            seek_proximity=True,
            show_sensor_plot=False,
            init_pos=np.array([-0.01, 0.8]),#davon sind motorcommands2 [0, 1.25]), #m3 schoener! [-0.01, 0.8]),
            gui=gui, sim=sim,
            modelfile='./compare_models/v_0.6_0.1' #'./saves/mode_T15_final.pt'
        )

        agents.append(a1)
        agents.append(a2)

        #obstacle = create_obstacles(1, gui, sim, positions=[np.array([0, 1.])], color=None, name=None)

        #a1.register_agents(a2)
        a2.register_agents([a1])

        a2.gui_att.show_target = False
        #a1.gui_att.show_predictions = False
        #a2.gui_att.show_predictions = False
        a1.gui_att.show_simulated_positions = False
        a2.gui_att.show_simulated_positions = False

        def set_targets(from_step, c_target):
            #for a in agents:
             #   print(a.data.sensors.get(from_step))

            # Target of agent 1 is random
            # Target of agent 2 is position of agent 1
            if (from_step == 0):
                # Create new target for the next target period plus the future prediction time steps
                print(int(a1.target_steps_total / TARGET_CHANGE_FREQUENCY))
                if fix_targets:
                    a1.create_target(1, NUM_TIME_STEPS, compare_targets[c_target])
                else:
                    a1.create_target(1, NUM_TIME_STEPS)
            else:
                if (a1.on_target_steps < 1000 and a1.target_steps_total < TARGET_CHANGE_FREQUENCY):
                    # Keep same target
                    target = a1.data.get_actinf_targets_block(from_step, from_step + 1)[0]
                    a1.create_target(1, NUM_TIME_STEPS, position=target)

                else:
                    # Create new target
                    print(int(a1.target_steps_total / TARGET_CHANGE_FREQUENCY))
                    if fix_targets:
                        a1.create_target(1, NUM_TIME_STEPS, compare_targets[c_target])
                    else:
                        a1.create_target(1, NUM_TIME_STEPS)
                    a1.target_steps_total = 0

            # Every time step
            a_pos = a1.data.positions.get(from_step)
            #print('position of A', a_pos)

            # Plot the distance of B to A
            b_pos = a2.data.positions.get(from_step)
            #print('position of B', b_pos)
            distance = np.linalg.norm(b_pos - a_pos)

            a2.performances = np.concatenate([a2.performances, [distance]])
            a2.plot.plot([distance])
            #comp_start.ex.m.plot(distance)
            # a1.plot.plot([target_steps_total], init_distance)

    if c.MODE == 11:
        agent_positions = np.array(
            [
                [-0.7, 0.85],
                [0.7, 0.85]
            ])
        target_positions = np.array(
            [
                [1.0, 0.85],
                [-1.0, 0.85]
            ])

        a_scenario = 0
        b_pos = np.array([0., 1.])

        a1 = Agent(id='A', color='red', init_pos=agent_positions[a_scenario], gui=gui, sim=sim, modelfile='./saves/mode_B_final.pt')
        a1.gui_att.show_simulated_positions = False
        a1.gui_att.show_predictions = False

        a2 = Agent(id='B', color='lightgreen', init_pos=b_pos, gui=gui, sim=sim, modelfile='./saves/mode_B_final.pt')
        a2.gui_att.show_simulated_positions = False
        a2.gui_att.show_target = False
        a2.gui_att.show_scv_targets = False
        a2.gui_att.show_predictions = False

        agents.append(a1)
        agents.append(a2)

        # B can see A
        a2.register_agents(a1)
        a1.register_agents(a2)


        def set_targets(from_step):
            nonlocal a_scenario

            if(from_step == 0):
                # Create new target for the next target period plus the future prediction time steps
                a1.create_target(1, NUM_TIME_STEPS, position=target_positions[a_scenario])
                a2.create_target(1, NUM_TIME_STEPS, position=b_pos)

            else:
                if(a1.on_target_steps < 1 and a1.target_steps_total < TARGET_CHANGE_FREQUENCY):
                    # Keep same target
                    a_target = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    a1.create_target(1, NUM_TIME_STEPS, position=a_target)
                    b_target = a2.data.get_actinf_targets_block(from_step, from_step+1)[0]
                    a2.create_target(1, NUM_TIME_STEPS, position=b_target)

                else:
                    # Create new target for the next target period plus the future prediction time steps
                    a_scenario = (a_scenario + 1) % len(agent_positions)

                    if a_scenario == 0:
                        filename = "actinf_stage" + get_stage() + "_agent" + str(a1.id) + "_mode" + str(c.MODE)
                        a1.plot.save(filename)
                        np.savetxt("./results/"+filename+".csv", np.asarray(a1.performances2D)[:, -1, :], delimiter=";")
                        sys.exit(0)

                    a_new_pos = agent_positions[a_scenario]
                    a1.data.reset(from_step)
                    a1.data.positions.write(np.array([a_new_pos]), from_step)
                    a1.initial_state = torch.stack(a1.net.init_hidden()).data.numpy()[:, -1, :, :]
                    a1.actinf_position_predictions = []
                    a1.actinf_sensor_predictions = []
                    a1.actinf_inputs = []

                    b_new_pos = b_pos
                    a2.data.reset(from_step)
                    a2.data.positions.write(np.array([b_new_pos]), from_step)
                    a2.initial_state = torch.stack(a2.net.init_hidden()).data.numpy()[:, -1, :, :]
                    a2.actinf_position_predictions = []
                    a2.actinf_sensor_predictions = []
                    a2.actinf_inputs = []

                    a1.create_target(1, NUM_TIME_STEPS, position=target_positions[a_scenario])
                    a2.create_target(1, NUM_TIME_STEPS, position=b_pos)

                    # Calculate the initial distance of the new target to the agent
                    my_pos = a1.data.positions.get_relative(0)
                    init_distance = np.linalg.norm(my_pos - target_positions[a_scenario])
                    print('Initial distance to target =',str(init_distance))

                    a1.on_target_steps = 0
                    a1.target_steps_total = 0

            # Plot performance
            A_pos = a1.data.positions.get_relative(0)
            B_pos = a2.data.positions.get_relative(0)
            #print(A_pos, B_pos)
            A_target_pos = a1.data.get_actinf_targets_block(from_step, from_step+1)[0]

            # Don't plot if the position of A is not set yet -> so if it's all zero
            if not np.any(A_pos):
                return

            # Distance of A to target
            target_dist = np.linalg.norm(A_pos - A_target_pos)

            # Distance of A to B
            agent_dist = np.linalg.norm(A_pos - B_pos)
            real_agent_dist = agent_dist - (2 * a1.radius)

            a1.plot.plot([target_dist, real_agent_dist])
            a1.performances2D.append([[target_dist, real_agent_dist]])



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
    position_counter = 0
    for t in range(NUM_ALL_STEPS):

        from_step = t
        to_step = from_step + NUM_TIME_STEPS

        comp_target = int(t/TARGET_CHANGE_FREQUENCY)

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
                    a.predict_future(from_step, input_t)

            # Prediction into future is done, now calculate error, perform bwpass
            # and apply gradients to motor commands
            for a in agents:
                velinf_needed = (c.MODE is 10 and a.id is 'B') or c.MODE is 82
                if velinf_needed:
                    if one_step_actinf: velinf_needed = False

                if velinf_needed:
                    # Perform velocity inference:
                    # Use the sensor loss to not adapt the motor commands (does not work well),
                    # but to adapt the velocities that would be needed to follow the gradient.
                    # These are then written to data.actinf_targets.
                    a.actinf(from_step, to_step, 1, velinf=True)
                    B_pos = a.data.positions.get(from_step)
                    #print('position of B', B_pos)

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
                vel_inf_clamp = (c.MODE is 10 and a.id is 'B')

                #without vel_inf:
                vel_inf_clamp = False

                if vel_inf_clamp:
                    a.actinf(from_step, to_step, LEARNING_RATE, vel_clamp=True)
                else:
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
                action = a.real_step(from_step, need_action=True)
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

        #if c.TAKE_SCREENSHOT is True:
            #gui.save_screenshot("{0:4}".format(from_step).replace(" ", "0"))

        # agents[0].sensorplot.save("{0:4}".format(from_step).replace(" ", "0"))
    # End

    for a in agents:
        filename = "chasing_smaller" + get_stage() + "_agent" + str(a.id) + "_mode" + str(c.MODE)
        # a.plot.save(filename)
        np.savetxt("./results/" + filename + ".csv", np.asarray(a.performances), delimiter=";")

    s.summary()
    print(data_capture)
    #np.save('motorcommands', data_capture)
    torch.save(data_capture , 'commands3.pt')


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
            name = 'O'+str(obstacle_index)
        
        o = Agent(id=name, color=color, init_pos=position, lr=None , gui=gui, sim=sim, is_obstacle=True)

        obstacles.append(o)
        o.gui_att.show_target = False

    return obstacles


compare_targets = []
positions = [[0,0.5],[0.5,0.5], [0.5, 1],[0.5, 1.5], [0, 1.5], [-0.5, 1.5], [-0.5, 1], [-0.5,0.5]]
position_counter = 0
for pos in positions:
    compare_targets.append(np.array(pos))
actinf()