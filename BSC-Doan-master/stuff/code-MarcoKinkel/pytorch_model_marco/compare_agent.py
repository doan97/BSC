import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import random
import time

import matplotlib.pyplot as plt

import global_config as c
from compare_simulator import Simulator, point_spread
from data import Data
from Gui_Attributes import Gui_attributes
from Net import Net
from plot import Plot
from circularPlot import SensorPlot
from compare_data_manipulator import Manipulator


# object unnötig
class Agent(object):
    def __init__(
            self,
            id,
            sim,
            init_pos,
            gui,
            num_epochs=100,
            modelfile=None,
            lr=None,
            color=None,
            radius=0.06,
            is_obstacle=False,
            stopwatch=None,
            position_loss_weight_actinf=c.POSITION_LOSS_WEIGHT_ACTINF,
            sensor_loss_weight_actinf=c.SENSOR_LOSS_WEIGHT_ACTINF,
            seek_proximity=False,
            show_sensor_plot=c.SHOW_SENSOR_PLOT,
            clamp_target_velocity_value=c.CLAMP_TARGET_VELOCITY_VALUE,
            v_clamp_target_velocity_value=c.CLAMP_TARGET_VELOCITY_VALUE_VELINF,
            v_noise=None,
            v_drop=None

    ):
        self.id = id
        self.data = Data(sim)

        self.sim = sim

        self.manipulator = Manipulator(velocity_noise=v_noise, velocity_dropout=v_drop)

        self.data.positions.append_single(init_pos)

        # Initialize GUI attributes and register the agent
        self.gui = None
        self.gui_att = None

        if gui is not None:
            self.gui = gui
            self.gui_att = Gui_attributes(self.gui, sim=sim, radius=radius, color=color)
            self.gui.register(self)

        # Create and load trained model, if given
        if False:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            name = torch.cuda.get_device_name(torch.cuda.current_device())

            self.net = Net(c.INPUT_DIM, c.HIDDEN_DIM, c.OUTPUT_DIM).to(device)
        else:
            self.net = Net(c.INPUT_DIM, c.HIDDEN_DIM, c.OUTPUT_DIM)

        if modelfile is not None:
            self.net.load_state_dict(torch.load(modelfile))

        # Node for Mean Squared Error
        self.mse = nn.MSELoss()
        self.mse2 = nn.MSELoss()

        # Optimizer only for learning, not active inference
        if lr is not None:
            self.optimiser = optim.Adam(self.net.parameters(), lr=lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, mode='min', factor=0.5, patience=10,
                                                                  verbose=True, threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)

        # Initialize hidden states
        previous_state = self.net.init_hidden()

        # Initialize simulation attributes
        self.other_agents = []
        self.radius = radius
        self.mass = 0.1
        self.init_pos = init_pos

        self.previous_scv = np.zeros(2)

        if self.gui_att is not None:
            self.gui_att.update_position(init_pos, np.zeros([c.INPUT_MOTOR_DIM]))

        self.sensor_directions = calc_sensor_dirs(c.INPUT_SENSOR_DIM)

        # These are cleared after every epoch
        self.losses = []
        self.losses_positions = []
        self.losses_sensors = []
        self.losses_accelerations = []

        # These are never cleared. They contain the mean losses for each epoch
        self.mean_losses = np.array([])
        self.mean_losses_positions = np.array([])
        self.mean_losses_sensors = np.array([])
        self.mean_losses_accelerations = np.array([])

        # For actinf
        self.performances = []
        self.performances2D = []

        self.show_sensor_plot = show_sensor_plot

        if is_obstacle is False:

            plot_xlims = None
            plot_ylims = None

            if modelfile is None:
                # Only use this kind of plot for learning

                plot_titles = ['Total loss', 'Position loss']
                plot_ylims = [.4, .2]
                plot_xlims = [num_epochs, num_epochs]

                if c.OUTPUT_SENSOR_DIM > 0:
                    plot_titles.append('Sensor loss')
                    plot_ylims.append(0.2)
                    plot_xlims.append(num_epochs)

                if c.OUTPUT_ACCELERATION_DIM > 0:
                    plot_titles.append('Acceleration loss')
                    plot_ylims.append(.2)
                    plot_xlims.append(num_epochs)

                linetype = '-'

            else:
                if c.MODE >= 70 and c.MODE <= 79:
                    plot_titles = ['Target distance', 'Agent distance']
                    plot_ylims = [2., 2.]
                    plot_xlims = None
                    linetype = '.'

                elif c.MODE == 9:
                    plot_titles = ['Relative distance to target', 'Distance to closest obstacle']
                    plot_ylims = [1., 2.]
                    plot_xlims = None
                    linetype = '.'

                else:
                    # use this plot for actinf
                    plot_titles = ['performance']
                    plot_ylims = None
                    plot_xlims = None
                    linetype = '.'

            self.plot = Plot(titles=plot_titles, ylims=plot_ylims, xlims=plot_xlims, title=id, linetype=linetype)

            if c.INPUT_SENSOR_DIM > 0 and show_sensor_plot is True:
                # print('graaaaa')
                self.sensorplot = SensorPlot(title=self.id, size=c.INPUT_SENSOR_DIM, color=color)

        self.s = stopwatch

        self.position_loss_weight_actinf = position_loss_weight_actinf
        self.sensor_loss_weight_actinf = sensor_loss_weight_actinf
        self.seek_proximity = seek_proximity

        self.on_target_steps = 0
        self.target_steps_total = 0

        self.path_scenario = None

        self.clamp_target_velocity_value = clamp_target_velocity_value
        self.v_clamp_target_velocity_value = v_clamp_target_velocity_value

    def register_agents(self, agents):
        """Adds agents to the list of known agents

        Parameters:
        agents -- single agent or list of agents to be added
        """

        if isinstance(agents, type([])):
            for a in agents:
                self.other_agents.append(a)

        else:
            self.other_agents.append(agents)

    def create_target(self, change_frequency, num_time_steps, position=None):
        """Creates a new target position and adds it to the actinf_targets-buffer in data-object
        for the given number of time steps

        Parameters:
            change_frequency -- number of time steps that the target is to be approached
            num_time_steps -- number of time steps that the agents predicts into the future
            position (default: None) -- fixed position of the new target. If None, the position is random

        Returns:
            Position of new target (2,)
        """
        # Create new targets and write to data-object
        if position is None:
            new_target = self.data.create_actinf_targets(change_frequency, num_time_steps)

        else:
            new_target = self.data.set_actinf_targets(position, change_frequency, num_time_steps)


        # Update GUI parameters
        if self.gui_att is not None:
            self.gui_att.update_target(new_target)

        return new_target

    def create_target_line(self, change_frequency, num_time_steps):
        # Create new targets and write to data-object
        new_target = self.data.create_actinf_targets_line(change_frequency, num_time_steps)

        return new_target

    '''
    Predict the own movements and sensor inputs in the future when performing given motor commands.
    '''

    def predict_future(self, from_step, input_t, no_sim=False, no_sim_command=None):
        # Calculate the sensor inputs for future time steps
        # Either simulate them or use the predictions from last time step
        # Simulation only works if the position of other agents is already
        # calculated for future time steps

        # Simulate the sensor data by using the positions of all other agents

        if c.INPUT_SENSOR_DIM > 0:
            sensor_data = self.sim.calc_sensor_data(from_step + input_t, from_step + input_t + 1, self)

            # Weight the calculated sensor data with the sensitivity
            # The predicted sensor data is weighted below
            if c.USE_SENSOR_SENSITIVITY:
                # Apply sensor sensitivity depending on direction
                delta_position = self.data.position_deltas.get(from_step + input_t)

                target_abs_pos = self.data.get_actinf_targets_block(from_step + input_t, from_step + input_t + 1)
                target_delta_position = target_abs_pos - self.data.positions.get(from_step + input_t)

                sensor_sensitivities_velocity = calc_sensor_sensitivities(self.sensor_directions, delta_position)
                sensor_sensitivities_target = calc_sensor_sensitivities(self.sensor_directions,
                                                                        target_delta_position[-1])

                sensor_sensitivities = (sensor_sensitivities_velocity + sensor_sensitivities_target) / 2.

                sensor_data = sensor_sensitivities * sensor_data

            # No matter if sensitivity is active or not: store the calculated sensor data
            self.data.sensors.write(sensor_data, from_step + input_t)

        # Get the input data for one time step, v and m
        input_data = self.get_input_data(from_step + input_t, from_step + input_t + 1)

        # The prospective inference takes its own predictions as input
        if (input_t > 0):

            position_delta_prediction = self.actinf_position_predictions[input_t - 1]
            sensor_prediction = self.actinf_sensor_predictions[input_t - 1]

            position_delta_indices = (
            torch.tensor(0), torch.tensor(0), torch.tensor(np.array(range(c.POSITION_DIM_START, c.POSITION_DIM_END))))
            sensor_indices = (
            torch.tensor(0), torch.tensor(0), torch.tensor(np.array(range(c.SENSOR_DIM_START, c.SENSOR_DIM_END))))

            with torch.no_grad():
                input_data.index_put_(position_delta_indices, position_delta_prediction)

                if c.OUTPUT_SENSOR_DIM > 0 and c.INPUT_SENSOR_DIM > 0:
                    input_data.index_put_(sensor_indices, sensor_prediction)

        # Set initial state
        self.net.hidden = self.actinf_previous_state

        # FWPass: predict path for current motor activity for one time step
        prediction, state = self.net.forward(input_data)

        prediction = prediction.view(-1)
        position_prediction = prediction[c.OUTPUT_POSITION_DIM_START:c.OUTPUT_POSITION_DIM_END]
        sensor_prediction = prediction[c.OUTPUT_SENSOR_DIM_START:c.OUTPUT_SENSOR_DIM_END]

        # In sensor-prediction mode, set the sensory predictions to 0 if there
        # was no real sensor input in the previous time step
        if c.OUTPUT_SENSOR_DIM > 0:
            if (input_t == 0):
                # We are in the first time step of prospective inference.
                # Get the previous sensor data from the last real step
                previous_sensor_input = self.data.sensors.get(from_step)
            else:
                previous_sensor_input = self.actinf_sensor_predictions[input_t - 1].data.numpy()

            # Erase the sensor prediction if previous sensor input was zero
            if np.array_equiv(previous_sensor_input, np.zeros(c.OUTPUT_SENSOR_DIM)):
                # Previous sensor input was completely 0, so don't predict any sensor input
                # in the current time step
                sensor_prediction = torch.tensor(np.zeros(c.OUTPUT_SENSOR_DIM, dtype=np.float32))

            if c.USE_SENSOR_SENSITIVITY:
                sensor_sensitivities = np.ones([c.INPUT_SENSOR_DIM])
                delta_position = position_prediction.data.numpy()

                if not np.linalg.norm(delta_position) == 0.:
                    for i in range(c.INPUT_SENSOR_DIM):
                        direction = self.sensor_directions[i]
                        sensor_sensitivities[i] = 0.5 * (np.dot(direction, delta_position) /
                                                         (np.linalg.norm(direction) * np.linalg.norm(
                                                             delta_position)) + 1.)

                sensor_prediction = torch.mul(sensor_prediction, torch.tensor(sensor_sensitivities, dtype=torch.float))

        # Summarize all predictions in a packed Tensor, keeping grad-information
        self.actinf_position_predictions.append(position_prediction)
        self.actinf_sensor_predictions.append(sensor_prediction)
        # inputs.append(input_data[-1, :, :])
        self.actinf_inputs.append(input_data)

        # Store the lstm state to use it as lstm input for the next time step
        self.actinf_previous_state = state

        # Pass the input through the simulator
        motor_commands = input_data[0, 0, c.MOTOR_DIM_START:c.MOTOR_DIM_END].data.numpy()
        previous_sim_position = self.data.positions.get(from_step + input_t)
        previous_sim_velocity = self.data.velocities.get(from_step + input_t)

        if no_sim:
            position_delta, position, velocity, acceleration = self.sim.update(previous_sim_velocity,
                                                                               previous_sim_position,
                                                                               no_sim_command, self,
                                                                               from_step + input_t)

        else:
            position_delta, position, velocity, acceleration = self.sim.update(previous_sim_velocity,
                                                                               previous_sim_position,
                                                                                motor_commands, self,
                                                                               from_step + input_t)

        # Append the simulated velocity and position to data-buffer.
        # These are partly used as lstm-input for the next prospective inference steps
        # and as inputs for the simulator in the next step.
        self.data.position_deltas.append_single(position_delta)
        self.data.positions.append_single(position)

        self.data.accelerations.append_single(acceleration)

        # Store the velocity of the agent as input for the next call of update
        self.data.velocities.append_single(velocity)

        return input_data

    '''
    Until here the agent has predicted its own movements and sensor inputs in the future.
    Now it uses these predictions to optimize the motor commands to get closer to the target.
    '''

    def actinf(self, from_step, to_step, learning_rate, velinf=False, vel_clamp=False):
        num_time_steps = to_step - from_step
        inputs = torch.stack(self.actinf_inputs)[:, -1, -1, :].data.numpy()
        predictions_delta_position = torch.stack(self.actinf_position_predictions)

        # Calculate absolute prediction
        abs_pos = self.data.positions.get(from_step)
        predictions_abs_position = torch.tensor(abs_pos, dtype=torch.float) + torch.cumsum(
            torch.stack(self.actinf_position_predictions), dim=0)

        # Update GUI attributes
        if self.gui_att is not None:
            # GUI red line
            predictions_abs_position_np = predictions_abs_position.data.numpy()
            self.gui_att.update_actinf_path(predictions_abs_position_np)

            # GUI black line
            if self.gui_att.show_simulated_positions:
                previous_position = self.data.positions.get(from_step)
                previous_velocity = self.data.velocities.get(from_step)
                motor_commands = torch.stack(self.actinf_inputs).data.numpy()[:, -1, -1,
                                 c.MOTOR_DIM_START:c.MOTOR_DIM_END]  # 2:6

                _, real_positions, _ = self.sim.simulate_multiple(previous_position, previous_velocity, motor_commands,
                                                                  self, from_step)
                self.gui_att.update_simulated_positions_path(real_positions)

        # Get targets
        targets_abs_position = self.data.get_actinf_targets_block(from_step, to_step)
        # targets_abs_position = torch.tensor(targets_abs_position, dtype=torch.float32)

        # Calculate goal position deltas
        targets_delta_position = targets_abs_position - predictions_abs_position.data.numpy()

        if c.INPUT_SENSOR_DIM > 0:
            # Mask the targets for a specific amount of time to enable
            # planning around an obstacle
            if c.MASK_GRADIENTS_AT_PROXIMITY is True:
                first_sensor_data = inputs[0, c.SENSOR_DIM_START:c.SENSOR_DIM_END]
                if first_sensor_data.max() > 0.6:
                    for t in range(num_time_steps - 1):
                        targets_delta_position[t] = torch.tensor(np.zeros(2))

        # Clamp the target velocity
        # After SCV computation, otherwise the SCV addition would increase the length
        # of the velocity vector again
        if c.CLAMP_TARGET_VELOCITY:
            vel_length = np.linalg.norm(targets_delta_position, axis=1)
            if vel_clamp:
                clip_value = self.v_clamp_target_velocity_value
            else:
                clip_value = self.clamp_target_velocity_value

            for t in range(len(vel_length)):
                if vel_length[t] < -clip_value:
                    targets_delta_position[t] = -clip_value * targets_delta_position[t] / np.linalg.norm(
                        targets_delta_position[t])
                if vel_length[t] > clip_value:
                    targets_delta_position[t] = clip_value * targets_delta_position[t] / np.linalg.norm(
                        targets_delta_position[t])

        # Calculate loss to target
        targets_delta_position = torch.tensor(targets_delta_position, dtype=torch.float)
        # position_loss = self.mse(predictions_abs_position, targets_abs_position)
        position_loss = self.mse(predictions_delta_position, targets_delta_position)

        # Calculate sensor loss. The agents wants all distance sensors to be 0
        # We use the sensor predictions and not the real sensor data here,
        # since the agent wants to plan its future motor data by its predicted sensory input
        if c.OUTPUT_SENSOR_DIM > 0:
            sensor_predictions = torch.stack(self.actinf_sensor_predictions)

            # Calculate additional sensor loss.
            # If sensor loss should not be used, just set self.position_loss_weight to 1.0
            if self.seek_proximity is True:
                # The target for the sensors is high activity
                # sensor_targets = torch.ones_like(sensor_predictions)

                # Set sensor with max activity should be one. The others 0? or .5?
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
                # print('asdliasdhaisud',sensor_targets)

            else:
                # The target for the sensors is low activity
                sensor_targets = torch.zeros_like(sensor_predictions)

            sensor_loss = self.mse2(sensor_predictions, sensor_targets)

        else:
            sensor_loss = 0

        # Calculate total loss
        loss = self.position_loss_weight_actinf * position_loss + self.sensor_loss_weight_actinf * sensor_loss

        # print(str(loss.item()) + " = " + str(self.position_loss_weight_actinf * position_loss.item()) + " + " + str(self.sensor_loss_weight_actinf * sensor_loss.item()))

        # GUI: Update error text of target error
        # self.gui_att.update_target_error(loss.item())

        if self.gui is not None:
            self.gui.draw()

        # Create optimizer
        # Since we only use the optimizer for one step, we always have to create
        # a new one for each time step.
        optimizer = torch.optim.Adam(self.actinf_inputs, lr=learning_rate)

        # Calculate gradients
        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()

        # mask the gradients
        if velinf is True:
            mask_gradients_(self.actinf_inputs, velinf=True)
        else:
            mask_gradients_(self.actinf_inputs, velinf=False)

        # TMP: store the original motorcommands
        # inputs_tensor = torch.stack(self.actinf_inputs)
        # previous_motor_commands = inputs_tensor.data.numpy()[:, -1, -1, c.MOTOR_DIM_START:c.MOTOR_DIM_END]

        # Optimize the inputs using gradients
        optimizer.step()

        # grads = np.asarray([self.actinf_inputs[i].grad.data.numpy()[-1, -1] for i in range(len(self.actinf_inputs))])
        # print(grads)

        inputs_tensor = torch.stack(self.actinf_inputs)

        if velinf is True:
            optimized_velocities_np = inputs_tensor.data.numpy()[:, -1, -1,
                                      c.POSITION_DIM_START:c.POSITION_DIM_END]  # 0:2
            optimized_velocities_np *= -1
            targets_abs_position = abs_pos + np.cumsum(optimized_velocities_np, axis=0)
            # print(optimized_velocities_np)
            # optimized_velocities_np *= -1
            # print('targets ', targets_abs_position)
            # print('abs_pos', abs_pos)

            self.data.actinf_targets.write(targets_abs_position, from_step)
            # self.data.actinf_targets.write(optimized_velocities_np, from_step)
            # new_target = self.data.set_actinf_targets(targets_abs_position, 1, 1)

            # print('target pos', targets_abs_position)

        # # End VELINF

        else:
            # if self.id == 'A':
            # print('position A', abs_pos)
            optimized_motor_commands = inputs_tensor.data.numpy()[:, -1, -1, c.MOTOR_DIM_START:c.MOTOR_DIM_END]  # 2:6

            # After applying gradients, clip the motorcommands between (0,1)
            optimized_motor_commands = np.clip(optimized_motor_commands, 0, 1)

            # TMP: Calculate the change in motor commands
            # motor_commands_change = previous_motor_commands - optimized_motor_commands

            # Update motor commands in data-object to retrieve them in get_input_data
            # Motor commands are written here instead of appended!
            self.data.motor_commands.write(optimized_motor_commands, from_step)
        [i.detach_() for i in self.actinf_inputs]

    '''
    Until here the agent has predicted its own movements and sensor inputs in the future
    and used these predictions to optimize the motor commands.
    In the real step, we now use the real position
    and real sensor input (because they are currently observable)
    and these inferred motor commands to perform the movement of one time step
    '''

    def real_step(self, from_step, fix_movements=False, need_action=False, command=None):

        if c.INPUT_SENSOR_DIM > 0:
            sensor_data = self.sim.calc_sensor_data(from_step, from_step + 1, self)
            self.data.sensors.write(sensor_data, from_step)

        # Get input data for one time step, [v, m, s, a]
        input_data = self.get_input_data(from_step, from_step + 1)
        if command is not None:
            co = 0
            input_data[0][0][2] = command[0]
            input_data[0][0][3] = command[1]
            input_data[0][0][4] = command[2]
            input_data[0][0][5] = command[3]
            #for i, data in enumerate(input_data[0][0][2:6]):
            #    input_data[0][0][i] = command[co]
            #    co += 1
        #print(input_data)

        # GUI: Show the real position
        abs_pos = self.data.positions.get(from_step)

        if self.gui_att is not None:
            self.gui_att.update_position(
                abs_pos,
                input_data.data.numpy()[-1, -1, c.INPUT_POSITION_DIM:c.INPUT_POSITION_DIM + c.INPUT_MOTOR_DIM],  # 2:6
                input_data[0, 0, c.SENSOR_DIM_START:c.SENSOR_DIM_END])

            self.gui.draw()

        # Set initial state
        self.net.hidden = self.data.states.get(from_step)

        # Predict path for current motor activity for one time step
        prediction, state = self.net.forward(input_data)

        # Write lstm state to collection to use it in next time step
        self.data.states.append_single(torch.stack(state).data.numpy()[:, -1, :, :])

        # Simulate path for current motor activity for one time step
        motor_data_np = input_data.data.numpy()[0, 0, c.MOTOR_DIM_START:c.MOTOR_DIM_END]
        previous_sim_position = self.data.positions.get(from_step)
        previous_sim_velocity = self.data.velocities.get(from_step)
        sim_position_delta, sim_position, velocity, sim_acceleration = self.sim.update(previous_sim_velocity,
                                                                                       previous_sim_position,
                                                                                       motor_data_np, self, from_step)

        # Write the simulated velocity to the data object to retrieve it in above call of "get_combined_inputs_block"
        self.data.position_deltas.append_single(sim_position_delta)
        self.data.positions.append_single(sim_position)
        self.data.accelerations.append_single(sim_acceleration)

        # Write the velocity of the agent to an own buffer as input for the next simulation
        self.data.velocities.append_single(velocity)
        if need_action:
            print('real_step() data:', input_data[0][0][2:6])
            return input_data[0][0][2:6]

    def reset_learning(self):
        self.net.zero_grad()
        self.data.reset()

        self.data.positions.append_single(self.init_pos)
        self.data.position_deltas.change_curr_idx(1)
        self.data.velocities.change_curr_idx(1)
        self.data.motor_commands.change_curr_idx(1)
        self.data.sensors.change_curr_idx(1)
        self.data.accelerations.change_curr_idx(1)

        # Get initial hidden LSTM states
        self.initial_state = self.net.init_hidden()

        # These are cleared after every epoch
        self.losses = []
        self.losses_positions = []
        self.losses_sensors = []
        self.losses_accelerations = []

    def learning_mini_epoch(self, from_step, to_step, num_time_steps):

        inputs = self.data.get_combined_inputs_block(from_step, to_step)

        targets = self.data.get_targets_block(from_step, to_step)
        inputs = inputs[:, np.newaxis, :]
        # print(inputs)

        #Dropout
        #target is the same, input is different
        #inputs = self.manipulator.velocity_noise(0.3, inputs)
        #inputs = self.manipulator.velocity_dropout(1, inputs)
        inputs = self.manipulator.manipulate(inputs)

        # Convert input and targets to tensors
        inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Set hidden state to the previous last hidden state
        self.net.hidden = self.initial_state

        # Apply forward pass to get the predictions
        predictions, last_state = self.net.forward(inputs)

        position_change_predictions = predictions[:, c.OUTPUT_POSITION_DIM_START:c.OUTPUT_POSITION_DIM_END]
        sensor_predictions = predictions[:, c.OUTPUT_SENSOR_DIM_START:c.OUTPUT_SENSOR_DIM_END]
        acceleration_predictions = predictions[:, c.OUTPUT_ACCELERATION_DIM_START:c.OUTPUT_ACCELERATION_DIM_END]

        # Just for GUI
        self.sensor_predictions = sensor_predictions
        self.last_position_predictions = position_change_predictions

        # [:] for all elements in list
        position_targets = targets[:, c.OUTPUT_POSITION_DIM_START:c.OUTPUT_POSITION_DIM_END]
        sensor_targets = targets[:, c.OUTPUT_SENSOR_DIM_START:c.OUTPUT_SENSOR_DIM_END]
        acceleration_targets = targets[:, c.OUTPUT_ACCELERATION_DIM_START:c.OUTPUT_ACCELERATION_DIM_END]

        # Apply direction dependent sensor sensitivity
        if c.LEARN_SENSOR_SENSITIVITY:
            sensor_sensitivities = np.ones([c.INPUT_SENSOR_DIM])

            if not np.linalg.norm(position_targets) == 0.:
                for i in range(c.INPUT_SENSOR_DIM):
                    direction = self.sensor_directions[i]
                    # print(direction, position_targets)

                    sensor_sensitivities[i] = ((np.dot(direction, position_targets) /
                                                (np.sqrt(np.dot(direction, direction)) *
                                                 np.sqrt(np.dot(position_targets, position_targets)))) + 1.) / 2.

            sensor_targets = sensor_targets * sensor_sensitivities

        # Set the last state to the next initial state
        self.initial_state = last_state

        # Calculate error
        loss_positions = c.POSITION_WEIGHT_LEARNING * self.mse(position_change_predictions, position_targets)
        loss_sensors = c.SENSOR_WEIGHT_LEARNING * self.mse(sensor_predictions,
                                                           sensor_targets) if c.OUTPUT_SENSOR_DIM > 0 else torch.tensor(
            0.)
        loss_accelerations = c.ACCELERATION_WEIGHT_LEARNING * self.mse(acceleration_predictions,
                                                                       acceleration_targets) if c.OUTPUT_ACCELERATION_DIM > 0 else torch.tensor(
            0.)

        loss = loss_positions + loss_sensors + loss_accelerations

        # Zero out gradient, otherwise they will accumulate between epochs
        self.optimiser.zero_grad()

        # Perform bwpass
        loss.backward()

        # Update parameters
        self.optimiser.step()

        # Store losses for visualization
        self.losses.append(loss.item())
        self.losses_positions.append(loss_positions.item())
        self.losses_sensors.append(loss_sensors.item())
        self.losses_accelerations.append(loss_accelerations.item())

        # --End of mini-epoch

    def save(self, modelfile):
        torch.save(self.net.state_dict(), modelfile)

    def get_input_data(self, from_step, to_step):
        input_data = self.data.get_combined_inputs_block(from_step, to_step)
        input_data = input_data[:, np.newaxis, :]
        input_data = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)

        return input_data


def calc_sensor_dirs(number_of_sensors):
    # Directions:
    #      3
    #    4 | 2
    # --5---|---1
    #    6 | 8
    #      7

    dirs = []
    for i in range(number_of_sensors):
        sensor_range_rad = (2 * np.pi) / number_of_sensors
        angle_rad = (i * sensor_range_rad + (i + 1) * sensor_range_rad) / 2.

        dirs.append(np.array([np.cos(angle_rad), np.sin(angle_rad)]))

        # # Draw SCV directions
        # plt.plot(np.asarray(dirs)[:,0], np.asarray(dirs)[:,1], 'bo')
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.show(block=True)

    dirs = [np.round(dir, 6) for dir in dirs]

    return dirs


def calc_sensor_sensitivities(sensor_directions, vector):
    sensor_sensitivities = np.ones([c.INPUT_SENSOR_DIM])

    if not np.linalg.norm(vector) == 0.:
        for i in range(c.INPUT_SENSOR_DIM):
            direction = sensor_directions[i]
            sensor_sensitivities[i] = 0.5 * (np.dot(direction, vector) /
                                             (np.linalg.norm(direction) * np.linalg.norm(vector)) + 1.)

    return sensor_sensitivities


def mask_gradients_(inputs, velinf=False):
    if velinf is True:
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

    mask = torch.tensor(mask)
    for i in inputs:
        i.grad = i.grad * mask


def mse(a, b):
    return np.mean((a - b) ** 2)
