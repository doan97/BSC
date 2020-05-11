import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import WORKING_MARCO.global_config as c
from WORKING_MARCO.Gui_Attributes import Gui_attributes
from WORKING_MARCO.Net import Net
from WORKING_MARCO.circularPlot import SensorPlot
from WORKING_MARCO.data import Data
from WORKING_MARCO.plot import Plot
from WORKING_MARCO.simulator import point_spread
global less_inputs

class Agent(object):
    def __init__(
            self,
            id_,
            sim,
            init_pos,
            gui,
            num_epochs=100,
            model_file=None,
            lr=None,
            color=None,
            radius=0.06,
            is_obstacle=False,
            stopwatch=None,
            position_loss_weight_action_inference=c.POSITION_LOSS_WEIGHT_ACTINF,
            sensor_loss_weight_action_inference=c.SENSOR_LOSS_WEIGHT_ACTINF,
            seek_proximity=False,
            show_sensor_plot=c.SHOW_SENSOR_PLOT,
            clamp_target_velocity_value=c.CLAMP_TARGET_VELOCITY_VALUE,
            v_clamp_target_velocity_value=c.CLAMP_TARGET_VELOCITY_VALUE_VELINF,
            less_inputs=False,
            motor_only=False,
            input_type='all'

    ):
        self.less_inputs = less_inputs
        self.hidden_states = []
        self.id = id_
        self.data = Data(sim)

        self.sim = sim
        self.input_type = input_type
        self.data.positions.append_single(init_pos)

        # Initialize GUI attributes and register the agent
        self.gui = None
        self.gui_att = None

        if gui is not None:
            self.gui = gui
            self.gui_att = Gui_attributes(self.gui, sim=sim, radius=radius, color=color)
            self.gui.register(self)

        # Create and load trained model, if given
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.input_type == 'motor only':
            self.net = Net(c.INPUT_MOTOR_DIM, c.HIDDEN_DIM, c.OUTPUT_DIM).to('cpu')
        elif self.input_type == 'motor and sensor':
            self.net = Net(c.INPUT_SENSOR_DIM + c.INPUT_MOTOR_DIM, c.HIDDEN_DIM, c.OUTPUT_DIM).to('cpu')
        elif self.input_type == 'all':
            self.net = Net(c.INPUT_DIM, c.HIDDEN_DIM, c.OUTPUT_DIM).to('cpu')
        self.net_sensor = Net(c.INPUT_DIM - 4, c.HIDDEN_DIM, c.OUTPUT_DIM).to('cpu') #without vel, pos as input

        self.net_motor_only = Net(4, c.HIDDEN_DIM, c.OUTPUT_DIM).to('cpu')

        #self.net_sensor.load_state_dict(self.net.state_dict())

        if model_file is not None:
            self.net.load_state_dict(torch.load(model_file))

        # Node for Mean Squared Error
        self.mse = nn.MSELoss()
        self.mse2 = nn.MSELoss()

        # Optimizer only for learning, not active inference
        if lr is not None:
            self.optimiser = optim.Adam(self.net.parameters(), lr=lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, mode='min', factor=0.5, patience=10,
                                                                  verbose=True, threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)

            self.optimiser2 = optim.Adam(self.net_sensor.parameters(), lr=lr)
            self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser2, mode='min', factor=0.5, patience=10,
                                                                  verbose=True, threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)

            self.optimiser3 = optim.Adam(self.net_motor_only.parameters(), lr=lr)
            self.scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser3, mode='min', factor=0.5, patience=10,
                                                                  verbose=True, threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)


        # Initialize hidden states
        previous_state = self.net.init_hidden()

        # because this code is a cluster fuck i do not know if initial state can be set as init hidden
        # or there is an usage later on that expects that initial state does not exist...
        self.initial_state = None

        self.sensor_predictions = None
        self.last_position_predictions = None

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
        self.losses2 = []
        self.losses3 = []
        #self.loss_net_sensor = []
        #self.loss_marco = []
        self.losses_positions = []
        self.losses_positions2 = []
        self.losses_positions3 = []
        self.losses_sensors = []
        self.losses_sensors2 = []
        self.losses_sensors3 = []
        self.losses_accelerations = []
        self.losses_accelerations2 = []
        self.losses_accelerations3 = []

        # These are never cleared. They contain the mean losses for each epoch
        self.mean_losses = np.array([])
        self.mean_losses_positions = np.array([])
        self.mean_losses_sensors = np.array([])
        self.mean_losses_accelerations = np.array([])

        # For action inference
        self.performances = []
        self.performances2D = []

        self.show_sensor_plot = show_sensor_plot

        if is_obstacle is False:

            if model_file is None:
                # Only use this kind of plot for learning

                plot_titles = ['Total loss', 'Position loss']
                plot_limit_y = [.4, .2]
                plot_limit_x = [num_epochs, num_epochs]

                if c.OUTPUT_SENSOR_DIM > 0:
                    plot_titles.append('Sensor loss')
                    plot_limit_y.append(0.2)
                    plot_limit_x.append(num_epochs)

                if c.OUTPUT_ACCELERATION_DIM > 0:
                    plot_titles.append('Acceleration loss')
                    plot_limit_y.append(.2)
                    plot_limit_x.append(num_epochs)

                line_type = '-'

            else:
                if 70 <= c.MODE <= 79:
                    plot_titles = ['Target distance', 'Agent distance']
                    plot_limit_y = [2., 2.]
                    plot_limit_x = None
                    line_type = '.'

                elif c.MODE == 9:
                    plot_titles = ['Relative distance to target', 'Distance to closest obstacle']
                    plot_limit_y = [1., 2.]
                    plot_limit_x = None
                    line_type = '.'

                else:
                    # use this plot for action inference
                    plot_titles = ['performance']
                    plot_limit_y = None
                    plot_limit_x = None
                    line_type = '.'

            self.plot = Plot(titles=plot_titles, ylims=plot_limit_y, xlims=plot_limit_x, title=id_, linetype=line_type)
            self.plot2 = Plot(titles=plot_titles, ylims=plot_limit_y, xlims=plot_limit_x, title=id_, linetype=line_type)
            self.plot3 = Plot(titles=plot_titles, ylims=plot_limit_y, xlims=plot_limit_x, title=id_, linetype=line_type)

            if c.INPUT_SENSOR_DIM > 0 and show_sensor_plot is True:
                # TODO there is no other occurrence of .sensor_plot in the corresponding project files
                self.sensor_plot = SensorPlot(title=self.id, size=c.INPUT_SENSOR_DIM, color=color)

        self.s = stopwatch

        self.position_loss_weight_action_inference = position_loss_weight_action_inference
        self.sensor_loss_weight_action_inference = sensor_loss_weight_action_inference
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

    def predict_future(self, from_step, input_t, com=None):
        # Calculate the sensor inputs for future time steps
        # Either simulate them or use the predictions from last time step
        # Simulation only works if the position of other agents is already
        # calculated for future time steps

        # Simulate the sensor data by using the positions of all other agents
        if c.INPUT_SENSOR_DIM > 0:
            sensor_data = self.sim.calc_sensor_data(from_step + input_t, from_step + input_t + 1, self,
                                                    self.data.sim.sensor_dirs, self.data.sim.borders)
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
        if input_t > 0:

            position_delta_prediction = self.actinf_position_predictions[input_t - 1]
            sensor_prediction = self.actinf_sensor_predictions[input_t - 1]

            position_delta_indices = (
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(np.array(range(c.POSITION_DIM_START, c.POSITION_DIM_END)))
            )
            sensor_indices = (
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(np.array(range(c.SENSOR_DIM_START, c.SENSOR_DIM_END)))
            )

            with torch.no_grad():
                input_data.index_put_(position_delta_indices, position_delta_prediction)

                if c.OUTPUT_SENSOR_DIM > 0 and c.INPUT_SENSOR_DIM > 0:
                    input_data.index_put_(sensor_indices, sensor_prediction)

        # Set initial state
        self.net.hidden = self.actinf_previous_state

        # FWPass: predict path for current motor activity for one time step
        if self.less_inputs:
            input_data = input_data[:,:, 2:-2]
            input_data = input_data.clone().detach().requires_grad_(True)

        # Convert input and targets to tensors
        #input_data = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)
        prediction, state = self.net.forward(input_data)
        self.hidden_states.append(state)


        prediction = prediction.view(-1)
        position_prediction = prediction[c.OUTPUT_POSITION_DIM_START:c.OUTPUT_POSITION_DIM_END]
        sensor_prediction = prediction[c.OUTPUT_SENSOR_DIM_START:c.OUTPUT_SENSOR_DIM_END]

        # In sensor-prediction mode, set the sensory predictions to 0 if there
        # was no real sensor input in the previous time step
        if c.OUTPUT_SENSOR_DIM > 0:
            if input_t == 0:
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
        #motor_commands = input_data[0, 0, :4].data.numpy()
        previous_sim_position = self.data.positions.get(from_step + input_t)
        previous_sim_velocity = self.data.velocities.get(from_step + input_t)

        if com is not None:
            motor_commands = com

        position_delta, position, velocity, acceleration = self.sim.update(previous_sim_velocity, previous_sim_position,
                                                                           motor_commands, self, from_step + input_t)

        # Append the simulated velocity and position to data-buffer.
        # These are partly used as lstm-input for the next prospective inference steps
        # and as inputs for the simulator in the next step.
        self.data.position_deltas.append_single(position_delta)
        self.data.positions.append_single(position)

        self.data.accelerations.append_single(acceleration)

        # Store the velocity of the agent as input for the next call of update
        self.data.velocities.append_single(velocity)

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
                motor_commands = \
                    torch.stack(self.actinf_inputs).data.numpy()[:, -1, -1, c.MOTOR_DIM_START:c.MOTOR_DIM_END]  # 2:6

                _, real_positions, _ = self.sim.simulate_multiple(previous_position, previous_velocity, motor_commands,
                                                                  self, from_step)
                self.gui_att.update_simulated_positions_path(real_positions)

        # Get targets
        targets_abs_position = self.data.get_actinf_targets_block(from_step, to_step)
        # targets_abs_position = torch.tensor(targets_abs_position, dtype=torch.float32)

        if c.MODE == 81:
            # To use an SCV a target position must be given.
            # Since there is no target in mode 81, take the current position as target
            predictions_abs_position = torch.tensor(targets_abs_position)

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

            if c.USE_SCV is True and self.seek_proximity is False:
                # Calculate counter vector for each sensor for each time step
                scvs = np.zeros([num_time_steps, c.INPUT_SENSOR_DIM, 2])

                # Apply scv on predicted sensor data, if possible. In stage 2, apply it on the simulated data
                if c.OUTPUT_SENSOR_DIM > 0:
                    sensor_data = torch.stack(self.actinf_sensor_predictions).data.numpy()
                else:
                    sensor_data = self.data.sensors.get(from_step, from_step + num_time_steps)

                # Calculate SCV for each time step and each sensor
                for t in range(num_time_steps):
                    for i in range(c.INPUT_SENSOR_DIM):
                        scvs[t, i] = (-1) * (sensor_data[t, i] ** c.SCV_BETA) * self.sensor_directions[i]

                # Calculate SCV over all sensors for each time step
                target_change = []
                for t in range(num_time_steps):
                    previous_scv = self.data.scv.get_relative(-1)

                    scv = c.SCV_SMOOTHING_FACTOR * previous_scv + (
                            1. - c.SCV_SMOOTHING_FACTOR) * c.SCV_WEIGHTING_FACTOR * np.sum(scvs[t], axis=0)
                    self.data.scv.append_single(scv)

                    target_change.append(scv)

                # Sensory Counter Vector

                # for t in range(num_time_steps):
                #    target_change.append(torch.tensor(self.data.scv.get(from_step + t) / 4.**(t-1), dtype=torch.float))

                target_change = torch.tensor(np.stack(target_change).astype(np.float32))
                targets_delta_position = torch.tensor(targets_delta_position, dtype=torch.float) + target_change
                targets_abs_position = torch.tensor(targets_abs_position, dtype=torch.float) + target_change

                # GUI: SCV Lines
                if self.gui_att is not None:
                    self.gui_att.update_scv_lines(targets_abs_position.data.numpy())

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

            else:
                # The target for the sensors is low activity
                sensor_targets = torch.zeros_like(sensor_predictions)

            sensor_loss = self.mse2(sensor_predictions, sensor_targets)

        else:
            sensor_loss = 0

        # Calculate total loss
        loss = self.position_loss_weight_action_inference * position_loss + self.sensor_loss_weight_action_inference * sensor_loss


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
            mask_gradients_(self.actinf_inputs, velinf=True, less_inputs=self.less_inputs)
        else:
            mask_gradients_(self.actinf_inputs, velinf=False)

        # TMP: store the original motorcommands
        # inputs_tensor = torch.stack(self.actinf_inputs)
        # previous_motor_commands = inputs_tensor.data.numpy()[:, -1, -1, c.MOTOR_DIM_START:c.MOTOR_DIM_END]

        # Optimize the inputs using gradients
        optimizer.step()
        # grads = np.asarray([self.actinf_inputs[i].grad.data.numpy()[-1, -1] for i in range(len(self.actinf_inputs))])

        inputs_tensor = torch.stack(self.actinf_inputs)

        if velinf is True:
            optimized_velocities_np = inputs_tensor.data.numpy()[:, -1, -1,
                                      c.POSITION_DIM_START:c.POSITION_DIM_END]  # 0:2
            # optimized_velocities_np *= -1
            targets_abs_position = abs_pos + np.cumsum(optimized_velocities_np, axis=0)

            self.data.actinf_targets.write(targets_abs_position, from_step)
            # self.data.actinf_targets.write(optimized_velocities_np, from_step)
            # new_target = self.data.set_actinf_targets(targets_abs_position, 1, 1)


        # # End VELINF

        else:
            optimized_motor_commands = inputs_tensor.data.numpy()[:, -1, -1, c.MOTOR_DIM_START:c.MOTOR_DIM_END]  # 2:6
            if self.less_inputs:
                optimized_motor_commands = inputs_tensor.data.numpy()[:, -1, -1,
                                           0:4]  # 2:6

            # After applying gradients, clip the motor commands between (0,1)
            optimized_motor_commands = np.clip(optimized_motor_commands, 0, 1)

            # TMP: Calculate the change in motor commands
            # motor_commands_change = previous_motor_commands - optimized_motor_commands

            # Update motor commands in data-object to retrieve them in get_input_data
            # Motor commands are written here instead of appended!
            self.data.motor_commands.write(optimized_motor_commands, from_step, that_wrong=True)

        [i.detach_() for i in self.actinf_inputs]
        return loss.item()

    '''
    Until here the agent has predicted its own movements and sensor inputs in the future
    and used these predictions to optimize the motor commands.
    In the real step, we now use the real position
    and real sensor input (because they are currently observable)
    and these inferred motor commands to perform the movement of one time step
    '''

    def predict(self, inputs, hidden=None):
        return self.net.forward(inputs, hidden_=hidden)


    def real_step(self, from_step, com=None):

        if c.INPUT_SENSOR_DIM > 0:
            sensor_data = self.sim.calc_sensor_data(from_step, from_step + 1, self,
                                                    self.data.sim.sensor_dirs,
                                                    self.data.sim.borders)
            self.data.sensors.write(sensor_data, from_step)

        # Get input data for one time step, [v, m, s, a]
        input_data = self.get_input_data(from_step, from_step + 1)
        motor_commands = input_data[0][2:6]


        # GUI: Show the real position
        abs_pos = self.data.positions.get(from_step)
        velocity = input_data.data.numpy()[-1, -1, 0 : c.INPUT_POSITION_DIM]
        if com is None:
            motor_com = input_data.data.numpy()[-1, -1, c.INPUT_POSITION_DIM:c.INPUT_POSITION_DIM + c.INPUT_MOTOR_DIM]
        else:
            motor_com = com
            #motor_com = np.array([1,0,0,0]) #links unten
            #motor_com = np.array([0,1,0,0]) #rechts unten
            #motor_com = np.array([0,0,1,0]) #links oben
            #motor_com = np.array([0,0,0,1]) #rechts oben

        if self.gui_att is not None:
            self.gui_att.update_position(
                abs_pos,
                motor_com,  # 2:6
                input_data[0, 0, c.SENSOR_DIM_START:c.SENSOR_DIM_END])

            self.gui.draw()

        # Set initial state
        self.net.hidden = self.data.states.get(from_step)

        # Predict path for current motor activity for one time step
        if self.less_inputs:
            input_data = input_data[:,:, 2:-2]
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

        return motor_com, velocity, sim_position

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

    def learning_mini_epoch(self, from_step, to_step):
        """
        improve the model by training on the experience of the last mini epoch

        Parameters
        ----------
        from_step : ...
        to_step : ...

        Returns
        -------

        """
        inputs = self.data.get_combined_inputs_block(from_step, to_step)
        sensor_net_input = np.array([np.array(inp[2:-2]) for inp in inputs])
        motor_net_input = np.array([np.array(inp[2:6]) for inp in inputs])
        targets = self.data.get_targets_block(from_step, to_step)

        inputs = inputs[:, np.newaxis, :]
        sensor_net_input = sensor_net_input[:, np.newaxis, :]
        motor_net_input = motor_net_input[:, np.newaxis, :]


        # Convert input and targets to tensors
        inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        sensor_net_input = torch.tensor(sensor_net_input, dtype=torch.float32, requires_grad=True)
        motor_net_input = torch.tensor(motor_net_input, dtype=torch.float32, requires_grad=True)

        targets = torch.tensor(targets, dtype=torch.float32)

        # Set hidden state to the previous last hidden state
        # self.net.hidden = self.initial_state
        # btw the .hidden is >>never<< used as reference on the right side of equal sign
        # therefor this has no impact at all
        # one could argue that the ring buffer for the states is not necessary as well

        # Apply forward pass to get the predictions
        predictions, last_state = self.net.forward(inputs)
        predictions_sensor_net, last_state = self.net_sensor.forward(sensor_net_input)
        predictions_motor_net, last_state = self.net_motor_only.forward(motor_net_input)

        position_change_predictions = predictions[:, c.OUTPUT_POSITION_DIM_START:c.OUTPUT_POSITION_DIM_END]
        position_change_predictions2 = predictions_sensor_net[:, c.OUTPUT_POSITION_DIM_START:c.OUTPUT_POSITION_DIM_END]
        position_change_predictions3 = predictions_motor_net[:, c.OUTPUT_POSITION_DIM_START:c.OUTPUT_POSITION_DIM_END]

        sensor_predictions = predictions[:, c.OUTPUT_SENSOR_DIM_START:c.OUTPUT_SENSOR_DIM_END]
        sensor_predictions2 = predictions_sensor_net[:, c.OUTPUT_SENSOR_DIM_START:c.OUTPUT_SENSOR_DIM_END]
        sensor_predictions3 = predictions_motor_net[:, c.OUTPUT_SENSOR_DIM_START:c.OUTPUT_SENSOR_DIM_END]

        acceleration_predictions = predictions[:, c.OUTPUT_ACCELERATION_DIM_START:c.OUTPUT_ACCELERATION_DIM_END]
        acceleration_predictions2 = predictions_sensor_net[:, c.OUTPUT_ACCELERATION_DIM_START:c.OUTPUT_ACCELERATION_DIM_END]
        acceleration_predictions3 = predictions_motor_net[:, c.OUTPUT_ACCELERATION_DIM_START:c.OUTPUT_ACCELERATION_DIM_END]


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

                    sensor_sensitivities[i] = ((np.dot(direction, position_targets) /
                                                (np.sqrt(np.dot(direction, direction)) *
                                                 np.sqrt(np.dot(position_targets, position_targets)))) + 1.) / 2.

            sensor_targets = sensor_targets * sensor_sensitivities

        # Set the last state to the next initial state
        # self.initial_state = last_state
        # this has no impact as well

        # Calculate error
        loss_positions = c.POSITION_WEIGHT_LEARNING * self.mse(position_change_predictions, position_targets)
        loss_positions2 = c.POSITION_WEIGHT_LEARNING * self.mse(position_change_predictions2, position_targets)
        loss_positions3 = c.POSITION_WEIGHT_LEARNING * self.mse(position_change_predictions3, position_targets)

        if c.OUTPUT_SENSOR_DIM == 0:
            loss_sensors = torch.tensor(0.)
        else:
            loss_sensors = \
                c.SENSOR_WEIGHT_LEARNING * self.mse(sensor_predictions, sensor_targets)
            loss_sensors2 = \
                c.SENSOR_WEIGHT_LEARNING * self.mse(sensor_predictions2, sensor_targets)
            loss_sensors3 = \
                c.SENSOR_WEIGHT_LEARNING * self.mse(sensor_predictions3, sensor_targets)

        if c.OUTPUT_ACCELERATION_DIM == 0:
            loss_accelerations = torch.tensor(0.)
        else:
            loss_accelerations = \
                c.ACCELERATION_WEIGHT_LEARNING * self.mse(acceleration_predictions, acceleration_targets)
            loss_accelerations2 = \
                c.ACCELERATION_WEIGHT_LEARNING * self.mse(acceleration_predictions2, acceleration_targets)
            loss_accelerations3 = \
                c.ACCELERATION_WEIGHT_LEARNING * self.mse(acceleration_predictions3, acceleration_targets)

        loss = loss_positions + loss_sensors + loss_accelerations
        loss2 = loss_positions2 + loss_sensors2 + loss_accelerations2
        loss3 = loss_positions3 + loss_sensors3 + loss_accelerations3

        #self.loss_net_sensor.append([loss_positions2.item(), loss_sensors2.item(), loss_accelerations2.item()])
        #self.loss_marco.append([loss_positions.item(), loss_sensors.item(), loss_accelerations.item()])

        # Zero out gradient, otherwise they will accumulate between epochs
        self.optimiser.zero_grad()
        self.optimiser2.zero_grad()
        self.optimiser3.zero_grad()

        # Perform back ward pass
        loss.backward()
        loss2.backward()
        loss3.backward()

        # Update parameters
        self.optimiser.step()
        self.optimiser2.step()
        self.optimiser3.step()

        # Store losses for visualization
        self.losses.append(loss.item())
        self.losses2.append(loss2.item())
        self.losses3.append(loss3.item())
        self.losses_positions.append(loss_positions.item())
        self.losses_positions2.append(loss_positions2.item())
        self.losses_positions3.append(loss_positions3.item())
        self.losses_sensors.append(loss_sensors.item())
        self.losses_sensors2.append(loss_sensors2.item())
        self.losses_sensors3.append(loss_sensors3.item())
        self.losses_accelerations.append(loss_accelerations.item())
        self.losses_accelerations2.append(loss_accelerations2.item())
        self.losses_accelerations3.append(loss_accelerations3.item())

    def save(self, model_file):
        torch.save(self.net.state_dict(), model_file + '_A')
        torch.save(self.net_sensor.state_dict(), model_file + '_B')
        torch.save(self.net_motor_only.state_dict(), model_file + '_C')

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

    directions = []
    for i in range(number_of_sensors):
        sensor_range_rad = (2 * np.pi) / number_of_sensors
        angle_rad = (i * sensor_range_rad + (i + 1) * sensor_range_rad) / 2.

        directions.append(np.array([np.cos(angle_rad), np.sin(angle_rad)]))

        # # Draw SCV directions
        # plt.plot(np.asarray(directions)[:,0], np.asarray(directions)[:,1], 'bo')
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.show(block=True)

    directions = [np.round(direction, 6) for direction in directions]

    return directions


def calc_sensor_sensitivities(sensor_directions, vector):
    sensor_sensitivities = np.ones([c.INPUT_SENSOR_DIM])

    if not np.linalg.norm(vector) == 0.:
        for i in range(c.INPUT_SENSOR_DIM):
            direction = sensor_directions[i]
            sensor_sensitivities[i] = 0.5 * (np.dot(direction, vector) /
                                             (np.linalg.norm(direction) * np.linalg.norm(vector)) + 1.)

    return sensor_sensitivities


def mask_gradients_(inputs, velinf=False, less_inputs=False):
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
    if False:
        mask = mask[2:-2]
    for i in inputs:
        i.grad = i.grad * mask


def mse(a, b):
    return np.mean((a - b) ** 2)
