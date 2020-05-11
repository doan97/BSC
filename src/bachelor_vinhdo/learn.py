import faulthandler
import time
import random


import numpy as np
import torch


import WORKING_MARCO.global_config as c
from WORKING_MARCO.agent import Agent
from WORKING_MARCO.gui import GUI
from WORKING_MARCO.simulator import Simulator
from WORKING_MARCO.stopwatch import Stopwatch


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def train():
    faulthandler.enable()

    # CONSTANTS
    num_epochs = 200  # Amount of epochs of size num_mini_epochs
    num_mini_epochs = 30  # Amount of mini-epochs of size num_time_steps
    num_time_steps = 15
    learning_rate = 0.01

    if c.VISUAL_LEARNING is True:
        gui = GUI()
    else:
        gui = None

    s = Stopwatch()

    sim = Simulator(mode=1, stopwatch=s)

    agents = []

    a2 = Agent(id_='S5', color='green', init_pos=np.array([-0.01, 0.8]), lr=learning_rate, gui=gui, sim=sim,
               num_epochs=num_epochs)
    agents.append(a2)  # [0., 1.4]

    #obstacles = create_obstacles(4, gui=gui, sim=sim,\
    #                             positions=np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]))
    obstacles = create_obstacles(1, gui=gui, sim=gui)
    for a in agents:
        a.register_agents(obstacles)

        if a.gui_att is not None:
            a.gui_att.show_target = False

    if gui is not None:
        gui.draw()

    # --------------
    # MAIN PROGRAM
    # --------------

    for epoch in range(1, num_epochs + 1):

        if epoch == num_epochs:
            c.VISUAL_LEARNING_STEP_BY_STEP = True

        # Clear gradients, simulator and data
        reset_agents(agents)
        reset_obstacles(obstacles)

        if gui is not None:
            gui.draw()

        for mini_epoch in range(num_mini_epochs):

            # Define the next from and to step
            from_step = mini_epoch * num_time_steps
            to_step = from_step + num_time_steps

            # Set obstacle agents
            scenario = mini_epoch % len(c.LEARNING_SCENARIOS)
            create_obstacle_path(obstacles, c.LEARNING_SCENARIOS[scenario], num_time_steps + 1)

            if epoch > 100 and epoch % 2 == 1:
                stand_still = True
            else:
                stand_still = False

            # Create random motor commands, get inputs and targets for next x timesteps
            create_inputs_and_targets(agents, from_step, to_step, stand_still)

            for a in agents:
                a.learning_mini_epoch(from_step, to_step)

            # Draw gui for each time step
            if gui is not None:
                for _ in agents:
                    update_gui(agents, obstacles, from_step, to_step, num_time_steps)

        if epoch % 1 == 0 and epoch < num_epochs:
            for a in agents:
                print_and_plot(epoch, a)
                print_and_plot2(epoch, a)
                print_and_plot3(epoch, a)

        # if epoch % 20 == 0 and epoch < num_epochs:
        #    for a in agents:
        #        a.save('./saves/mode_' + str(a.id) + '_epoch' + str(epoch) + '.pt')

        for a in agents:
            a.scheduler.step(a.losses[-1])
            a.scheduler2.step(a.losses2[-1])

        # --End of epoch

    # End of all Epochs
    # Plot
    for a in agents:
        print_and_plot(num_epochs, a, final=True)

    # Save model
    for a in agents:
        print('should be saved')
        a.save('./saves/mode_' + str(a.id) + '_final.pt')

    # Save mean losses
    for a in agents:
        np.savetxt("./results/agent" + str(a.id) + "_loss.csv", a.mean_losses, delimiter=";")
        np.savetxt("./results/agent" + str(a.id) + "_loss_pos.csv", a.mean_losses_positions, delimiter=";")
        np.savetxt("./results/agent" + str(a.id) + "_loss_sens.csv", a.mean_losses_sensors, delimiter=";")
        np.savetxt("./results/agent" + str(a.id) + "_loss_acc.csv", a.mean_losses_accelerations, delimiter=";")

        a.plot.save(a.id)

    s.summary()

    #assert agents[-1].mean_losses[-1] == 6.907955634718140371e-03, \
    #    '{}!={}'.format(agents[-1].mean_losses[-1], 6.907955634718140371e-03)
    print('this works still the same')


def update_gui(agents, obstacles, from_step, to_step, num_time_steps):
    if c.VISUAL_LEARNING_STEP_BY_STEP is True:
        # Update prediction line, position and simulation line
        for step in range(from_step, to_step):
            for a in agents:
                if step == from_step:
                    set_gui_attributes(a, from_step, to_step, predictions=a.last_position_predictions,
                                       motor_data=a.data.motor_commands.get(from_step))
                    a.gui.hide_first_line = True

                # Update only position
                abs_pos = a.data.positions.get(step)
                motor_data = a.data.motor_commands.get(step)
                sensor_data = a.data.sensors.get(step)
                a.gui_att.update_position(abs_pos, motor_data, sensor_data)

                a.gui.draw()

                # Draw sensor plot
                if c.INPUT_SENSOR_DIM > 0 and a.show_sensor_plot is True:
                    if step < to_step:
                        # TODO not sure about that
                        next_sensor_data = a.data.sensors.get(step + 1)
                        if c.OUTPUT_SENSOR_DIM > 0:
                            a.sensorplot.update(next_sensor_data, a.sensor_predictions[step % num_time_steps])
                        else:
                            a.sensorplot.update(next_sensor_data, [])

            for o in obstacles:
                set_gui_attributes(o, step, to_step, update_sim_positions=False)

            time.sleep(.1)

    else:
        for a in agents:
            set_gui_attributes(a, from_step, to_step, a.last_position_predictions)

        for o in obstacles:
            set_gui_attributes(o, from_step, to_step)


def set_gui_attributes(a, from_step, to_step, predictions=None, update_sim_positions=True,
                       motor_data=None):
    if motor_data is None:
        motor_data = np.zeros([c.INPUT_MOTOR_DIM])

    # GUI: Get the absolute position
    abs_pos = torch.tensor(a.data.positions.get(from_step), dtype=torch.float32)
    a.gui_att.update_position(abs_pos, motor_data)

    # GUI: Get the absolute prediction path
    if predictions is not None:
        with torch.no_grad():
            abs_predictions = abs_pos + torch.cumsum(predictions, dim=0)
        a.gui_att.update_actinf_path(abs_predictions)

    # GUI: Get the real positions path
    if update_sim_positions is True:
        with torch.no_grad():
            real_positions = a.data.positions.get(from_step, to_step)
        a.gui_att.update_simulated_positions_path(real_positions)

    a.gui.draw()


def create_inputs_and_targets(agents, from_step, to_step, stand_still=False):
    """
    Create the data for each time step, for each agent

    Parameters
    ----------
    agents : ...
    from_step : ...
    to_step : ...
    stand_still : ...

    Returns
    -------

    """
    for t in range(from_step, to_step):
        for a in agents:

            a.data.create_inputs(a, t, stand_still)

    # At last, calculate the velocity and position for the motor commands of the very last time step
    # since the velocity is used as target for learning.
    # E.g. with num_time_steps = 20 the loop creates velocity[0:19] and motordata[0:19]
    # The targets that function "get_targets_block" returns are velocity[1:20].
    # Thus we must calculate velocity[20] by executing motordata[19].
    # After that, we decrease the data-indices, so that the next call of create_inputs will again start at
    # writing to sensordata[20].
    for a in agents:
        a.data.create_inputs(a, to_step, final=True)
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
            print(positions)
            position = positions[obstacle_index]

        o = Agent(id_='O' + str(obstacle_index), color='gray', init_pos=position, lr=None, gui=gui, sim=sim,
                  is_obstacle=True)

        obstacles.append(o)

        if o.gui_att is not None:
            o.gui_att.show_target = False
            o.gui_att.show_simulated_positions = False

    return obstacles


def create_obstacle_path(obstacles, path_type, num_time_steps):
    for o in obstacles:

        positions = np.zeros([num_time_steps, c.INPUT_POSITION_DIM])

        # Get initial position
        # x between -1.5 and 1.5
        x = (np.random.rand() * 3) - 1.5
        # y between 0 and 2
        y = np.random.rand() * 2

        positions[0] = np.array([x, y])

        if path_type is 'static':
            vel_x = 0.
            vel_y = 0.

            velocities = np.array([vel_x, vel_y])

            # Get all positions of one epoch
            for t in range(1, num_time_steps):
                positions[t] = positions[t - 1] + velocities

        elif path_type is 'alone':
            for t in range(0, num_time_steps):
                positions[t] = np.array([1000., 1000.])

        elif 'line' in path_type:
            vel_x = np.random.rand() * c.CLAMP_TARGET_VELOCITY_VALUE
            vel_x -= c.CLAMP_TARGET_VELOCITY_VALUE / 2.
            vel_y = np.random.rand() * c.CLAMP_TARGET_VELOCITY_VALUE
            vel_y -= c.CLAMP_TARGET_VELOCITY_VALUE / 2.

            velocities = np.array([vel_x, vel_y])

            # Get all positions of one epoch
            for t in range(1, num_time_steps):
                positions[t] = positions[t - 1] + velocities

                if 'acc' in path_type:
                    velocities *= 1.05

        elif 'curve' in path_type:
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

                if 'acc' in path_type:
                    delta_angle *= 1.01

        elif 'designed' in path_type:
            for t in range(0, num_time_steps):
                positions[t] = np.array([1000., 1000.])

        o.data.positions.append(positions)
        o.data.positions.change_curr_idx(-1)


def print_and_plot(epoch, a, final=False):
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
    else:
        loss_sensors_mean = np.array([-1])

    if c.OUTPUT_ACCELERATION_DIM > 0:
        losses_accelerations = np.asarray(a.losses_accelerations)
        loss_accelerations_mean = np.mean(losses_accelerations)
        a.mean_losses_accelerations = np.append(a.mean_losses_accelerations, loss_accelerations_mean)
        plot_losses.append(loss_accelerations_mean)
    else:
        loss_accelerations_mean = np.array([-1])

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



def print_and_plot2(epoch, a, final=False):
    plot_losses = []
    f = "{0:0.7f}"

    losses = np.asarray(a.losses2)
    loss_mean = np.mean(losses)
    a.mean_losses2 = np.append(a.mean_losses, loss_mean)
    plot_losses.append(loss_mean)

    losses_positions = np.asarray(a.losses_positions2)
    loss_positions_mean = np.mean(losses_positions)
    a.mean_losses_positions2 = np.append(a.mean_losses_positions, loss_positions_mean)
    plot_losses.append(loss_positions_mean)

    if c.OUTPUT_SENSOR_DIM > 0:
        losses_sensors = np.asarray(a.losses_sensors2)
        loss_sensors_mean = np.mean(losses_sensors)
        a.mean_losses_sensors2 = np.append(a.mean_losses_sensors, loss_sensors_mean)
        plot_losses.append(loss_sensors_mean)
    else:
        loss_sensors_mean = np.array([-1])

    if c.OUTPUT_ACCELERATION_DIM > 0:
        losses_accelerations = np.asarray(a.losses_accelerations2)
        loss_accelerations_mean = np.mean(losses_accelerations)
        a.mean_losses_accelerations2 = np.append(a.mean_losses_accelerations, loss_accelerations_mean)
        plot_losses.append(loss_accelerations_mean)
    else:
        loss_accelerations_mean = np.array([-1])

    string = ''

    if not final:
        string += "E" + str(epoch)
    else:
        string += "END"

    string += "\t" + 'SN' + \
              "\tloss=" + str(f.format(loss_mean)) + \
              "\tl_pos=" + str(f.format(loss_positions_mean))

    if c.OUTPUT_SENSOR_DIM > 0:
        string += "\tl_sen=" + str(f.format(loss_sensors_mean))

    if c.OUTPUT_ACCELERATION_DIM > 0:
        string += "\tl_acc=" + str(f.format(loss_accelerations_mean))

    print(string)

    a.plot2.plot(plot_losses, persist=final)

def print_and_plot3(epoch, a, final=False):
    plot_losses = []
    f = "{0:0.7f}"

    losses = np.asarray(a.losses3)
    loss_mean = np.mean(losses)
    a.mean_losses3 = np.append(a.mean_losses, loss_mean)
    plot_losses.append(loss_mean)

    losses_positions = np.asarray(a.losses_positions3)
    loss_positions_mean = np.mean(losses_positions)
    a.mean_losses_positions3 = np.append(a.mean_losses_positions, loss_positions_mean)
    plot_losses.append(loss_positions_mean)

    if c.OUTPUT_SENSOR_DIM > 0:
        losses_sensors = np.asarray(a.losses_sensors3)
        loss_sensors_mean = np.mean(losses_sensors)
        a.mean_losses_sensors3 = np.append(a.mean_losses_sensors, loss_sensors_mean)
        plot_losses.append(loss_sensors_mean)
    else:
        loss_sensors_mean = np.array([-1])

    if c.OUTPUT_ACCELERATION_DIM > 0:
        losses_accelerations = np.asarray(a.losses_accelerations3)
        loss_accelerations_mean = np.mean(losses_accelerations)
        a.mean_losses_accelerations3 = np.append(a.mean_losses_accelerations, loss_accelerations_mean)
        plot_losses.append(loss_accelerations_mean)
    else:
        loss_accelerations_mean = np.array([-1])

    string = ''

    if not final:
        string += "E" + str(epoch)
    else:
        string += "END"

    string += "\t" + 'MT' + \
              "\tloss=" + str(f.format(loss_mean)) + \
              "\tl_pos=" + str(f.format(loss_positions_mean))

    if c.OUTPUT_SENSOR_DIM > 0:
        string += "\tl_sen=" + str(f.format(loss_sensors_mean))

    if c.OUTPUT_ACCELERATION_DIM > 0:
        string += "\tl_acc=" + str(f.format(loss_accelerations_mean))

    print(string)

    a.plot3.plot(plot_losses, persist=final)