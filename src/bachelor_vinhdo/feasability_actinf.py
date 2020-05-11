import bachelor_vinhdo.global_config as c
import numpy as np
import torch
import bachelor_vinhdo.utils as utils
import bachelor_vinhdo.draw as draw


from WORKING_MARCO.agent import Agent
from bachelor_vinhdo.simulator import Simulator
from WORKING_MARCO.stopwatch import Stopwatch
import bachelor_vinhdo.learn as learn

import torch.nn as nn

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

#Action inference
def action_inference(model_file, init_pos, target_position, obstacle_positions, timesteps=1, actinf_iterations=1, prediction_horizon=1, input_type='all', seek_proximity=False):
    gui = None
    s = Stopwatch()
    sim = Simulator(mode=1, stopwatch=s)
    agent = Agent(id_='A', color='red', init_pos=init_pos, gui=gui, sim=sim,
                   model_file=model_file, input_type=input_type)
    used_motor_commands = []

    positions = []
    sensor_information_of_all_steps = []
    velocities = [init_pos]
    accelerations = [np.array([0,0])]
    sensor_predictions_all = []
    seek_velocities_all = []
    all_states = []

    current_position = init_pos
    current_velocity = np.array([0,0])
    current_position_delta = np.array([0,0])
    current_acceleration = np.array([0,0])

    initialize_state = (torch.zeros(1, 1, 36), torch.zeros(1, 1, 36))
    current_state = initialize_state
    start_command = np.zeros(4)
    command_sequence = utils.get_command_sequence(prediction_horizon, start_command)

    obstacle_dicts = []
    for pos in obstacle_positions:
        obstacle_dicts.append({'position' : pos,\
                               'radius': 0.06})
    for t in range(timesteps):
        current_sensor_information = utils.get_sensor_data(agent, t, 0, current_position, obstacle_dicts)
        for ai in range(actinf_iterations):
            motor_commands, sensor_predictions, seek_velocities = actinf_iteration(agent,
                                              command_sequence,
                                              np.array([current_position]),
                                              np.array([target_position]),
                                              np.array([current_position_delta]),
                                              np.array([current_acceleration]),
                                              current_sensor_information,
                                              current_state,
                                              input_type=input_type,
                                              seek_proximity=seek_proximity)
            #shift motorcommands
            command_sequence = list(motor_commands[1:])
            #add a new pseudo random motorcommand, with the last command in sequence
            command_sequence.append(utils.get_random_motor_commands(command_sequence[-1]))
            command_sequence = np.array(command_sequence)[:, np.newaxis]

        next_motor_command = motor_commands[0]
        sensor_predictions = [pred.detach().numpy() for pred in sensor_predictions]
        sensor_predictions_all.append(sensor_predictions)
        if seek_proximity:
            seek_velocities_all.append(seek_velocities.detach().numpy())
        all_states.append([current_state[0].detach().numpy(), current_state[1].detach().numpy()])
        used_motor_commands.append(next_motor_command)
        #update current hidden, therefor forward through net
        if input_type == 'all':
            new_input = np.concatenate([np.array([current_position_delta]), np.array([next_motor_command])], axis=1)
            new_input = np.concatenate([new_input, current_sensor_information], axis=1)
            new_input = np.concatenate([new_input, np.array([current_acceleration])], axis=1)
        elif input_type == 'motor and sensor':
            new_input = np.concatenate([np.array([next_motor_command]), current_sensor_information], axis=1)
        elif input_type == 'motor only':
            new_input = np.array([next_motor_command])

        new_input = new_input[:, np.newaxis, :]
        new_input = torch.tensor(new_input, dtype=torch.float)

        pred, (h,c) = agent.net.forward(new_input, hidden_=current_state)
        current_state = (h.detach(), c.detach())

        current_position_delta, current_position, current_velocity, current_acceleration = \
            utils.simulate(sim, agent, next_motor_command, current_velocity, current_position, t, 0, obstacle_dicts)

        accelerations.append(current_acceleration)
        velocities.append(current_position_delta)
        positions.append(current_position)

        print('time step', t)
        sensor_information_of_all_steps.append(current_sensor_information)

    return positions, sensor_information_of_all_steps, used_motor_commands, velocities, accelerations, sensor_predictions_all, seek_velocities_all, all_states

def actinf_iteration(agent, command_sequence, init_pos, targets_pos, velocity, acceleration, sensor_information, hidden_state, input_type='all', seek_proximity=False):
    #coms = np.random.random((prediction_horizon,4))[:, np.newaxis]
    coms = command_sequence
    position_delta_predictions, sensor_predictions, inputs = predict(coms, agent, velocity, sensor_information, acceleration, hidden_state, input_type=input_type)
    position_delta_targets = get_velocity_delta_targets(init_pos, position_delta_predictions, targets_pos)

    sensor_targets = get_sensor_targets(sensor_predictions)
    motor_commands, optim_velocities = action_inference_step(position_delta_predictions, position_delta_targets, sensor_predictions, sensor_targets, inputs, input_type=input_type, seek_proximity=seek_proximity)
    return motor_commands, sensor_predictions, optim_velocities

def predict(coms, agent, position_delta, sensor_data, acceleration, hidden_state, input_type='all'):
    position_delta_tensors = []
    acceleration_tensors = []
    sensor_tensors = []
    position_delta_predictions =[position_delta]
    acceleration_predictions = [acceleration]
    sensor_predictions = [sensor_data]
    inputs = []
    forward_own_input = True

    hidden = [hidden_state]

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
        elif forward_own_input:
            if input_type == 'motor only':
                input_data = com
            elif input_type == 'motor and sensor':
                input_data = np.concatenate([com, np.array([sensor_predictions[-1].detach().numpy()])],axis=1)
            elif input_type == 'all':
                input_data = np.concatenate([np.array([position_delta_predictions[-1].detach().numpy()]), com], axis=1)
                input_data = np.concatenate([input_data, np.array([sensor_predictions[-1].detach().numpy()])], axis=1)
                input_data = np.concatenate([input_data, np.array([acceleration_predictions[-1].detach().numpy()])], axis=1)
            else:
                print('input type is not implemented')
                return
            input_data = input_data[:, np.newaxis, :]
            input_data = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)


        #TODO
        pred, (h, c) = agent.net.forward(input_data, hidden_=hidden[-1])
        #h.detach_()
        #c.detach_()
        #clamp states
        h.clamp_(-1, 1)
        c.clamp_(-1, 1)
        h *= 0.95
        c *= 0.95


        inputs.append(input_data)
        pred = pred.view(-1)
        hidden.append((h, c))
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
        optimized_velocities = optimize_velocities(input_type, inputs, learning_rate, sensor_loss)
        position_loss = mse(predictions_delta_position, optimized_velocities)
    else:
        position_loss = mse(predictions_delta_position, position_delta_targets)
        optimized_velocities = None

    optimized_motor_commands = optimize_motor_commands_vel_induced(input_type, inputs, learning_rate, position_loss)
    return optimized_motor_commands, optimized_velocities

def optimize_velocities(input_type, inputs, learning_rate, sensor_loss, sens_l=100):
    sensor_loss = sens_l * sensor_loss
    optimizer = torch.optim.Adam(inputs, lr=learning_rate, betas=(0, 0.9))
    optimizer.zero_grad()

    sensor_loss.backward(retain_graph=True)
    utils.mask_gradients_(inputs, input_type=input_type, velinf=True)
    optimizer.step()
    inputs_tensor = torch.stack(inputs)
    optimized_velocities = inputs_tensor.data.numpy()[:, -1, -1, :2]
    optimized_velocities = np.clip(optimized_velocities, 0, 1)
    #TESTING
    return torch.tensor(optimized_velocities, dtype=torch.float32)

def optimize_motor_commands_vel_induced(input_type, inputs, learning_rate, position_loss, vel_l=100):
    velocity_loss = vel_l * position_loss
    loss = velocity_loss
    optimizer = torch.optim.Adam(inputs, lr=learning_rate, betas=(0, 0.9))
    optimizer.zero_grad()
    loss.backward()
    utils.mask_gradients_(inputs, input_type=input_type)

    optimizer.step()
    inputs_tensor = torch.stack(inputs)
    if input_type == 'all':
        optimized_motor_commands = inputs_tensor.detach().numpy()[:, -1, -1, c.MOTOR_DIM_START:c.MOTOR_DIM_END]
    elif input_type == 'motor and sensor' or input_type == 'motor only':
        optimized_motor_commands = inputs_tensor.detach().numpy()[:, -1, -1, :4]

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
                point_spread_signal_tmp = utils.point_spread(sensor_index, sensor_targets[t, sensor_index],
                                                       c.POINT_SPREAD_TYPE)
                point_spread_signal_tmp[sensor_index] = 0
                point_spread_signal += point_spread_signal_tmp
            # Only after all point spread additions is calculated, add it to the sensor data
            sensor_targets[t] = sensor_targets[t] + point_spread_signal
            sensor_targets[t] = np.clip(sensor_targets[t], 0, 1)
            sensor_targets[t] = np.round(sensor_targets[t], decimals=8)

    sensor_targets = torch.tensor(sensor_targets)
    return sensor_targets

#drawing stuff

def run_that_shit(init_pos, target, obstacles, t=100, models=[True, True, True], seek=False, models_path='', folder=''):
    if models[0]:
        pos1, sensor_data1, used_coms1, vels1, accs1, sensor_pred1, seek_vels1, states1 = action_inference(
            models_path + 'model_A', np.array(init_pos), np.array(target), obstacles,
            timesteps=t, actinf_iterations=10,
            prediction_horizon=10, input_type='all', seek_proximity=seek)
        if seek:
            file_name = 'run_A_seek.npy'
        else:
            file_name = 'run_A.npy'
        np.save(folder + file_name, [pos1,
                                       sensor_data1,
                                       used_coms1,
                                       vels1,
                                       accs1,
                                       sensor_pred1,
                                       seek_vels1,
                                       states1])
    if models[1]:
        pos2, sensor_data2, used_coms2, vels2, accs2, sensor_pred2, seek_vels2, states2 = action_inference(
            models_path + 'model_B', np.array(init_pos), np.array(target), obstacles,
            timesteps=t, actinf_iterations=10,
            prediction_horizon=10, input_type='motor and sensor', seek_proximity=seek)
        if seek:
            file_name = 'run_B_seek.npy'
        else:
            file_name = 'run_B.npy'
        np.save(folder + file_name, [pos2,
                                       sensor_data2,
                                       used_coms2,
                                       vels2,
                                       accs2,
                                       sensor_pred2,
                                       seek_vels2,
                                       states2])
    if models[2]:
        pos3, sensor_data3, used_coms3, vels3, accs3, sensor_pred3, seek_vels3, states3 = action_inference(
            models_path + 'model_C', np.array(init_pos), np.array(target), obstacles,
            timesteps=t, actinf_iterations=10,
            prediction_horizon=10, input_type='motor only', seek_proximity=seek)
        if seek:
            file_name = 'run_C_seek.npy'
        else:
            file_name = 'run_C.npy'
        np.save(folder + file_name, [pos3,
                                       sensor_data3,
                                       used_coms3,
                                       vels3,
                                       accs3,
                                       sensor_pred3,
                                       seek_vels3,
                                       states3])

def run_all():
    obstacles = np.array([[-1., 1.5], [1., 1.5], [-1.,0.5], [1.,0.5]])
    #models = ['./models/with_border/moving_obstacles/', './models/with_border/no_obstacles/', './models/with_border/static_obstacles/',\
    models = ['./models/without_border/no_obstacles/', './models/without_border/static_obstacles/']
    for m in models:
        print(m)
        run_that_shit([0.,1.],[1., 1.8], obstacles, t=200, models_path=m,  folder=m)
        run_that_shit([0.,1.],[1., 1.8], obstacles, t=200, models_path=m,  folder=m, seek=True)
        print('neeext')
run_all()
#def show_that_shit(obstacles, target):
#    folder = './actinf_data/'
#    pos = np.load(folder + 'run_positions_A.npy')
#    sensor = np.load(folder + 'run_sensors_A.npy')

    #draw.draw_sensor_data(sensor)
    #print(sensor[10])
    #draw.draw_timestep(sensor, pos, obstacles, 100)
#    draw.draw_path(pos, [target], obstacles)

#def show_that_sensor_induced_shit(obstacles):#not moving obstacles
#    live_gui = draw.LiveGui('./actinf_data/run_A.npy', obstacles)
#    #live_gui.start('run_A.npy')


#run_that_shit([0.,1.],[1., 1.8], [[-1., 1.5]], t=200, models=[True,False,False], seek=True)
#show_that_shit([[-1., 1.5], [1., 1.5], [-1.,0.5], [1.,0.5]], [1., 1.8])
#show_that_sensor_induced_shit([[-1., 1.5]])

#TODO motorcommands pseudo random           done
# TODO sensorik testen          done
#TODO Hidden/ cell states clipen            done
#TODO motorkommandos shiften (nicht immer neu generieren)           done
#TODO plots erstellen done
# von: position prediction (simulator, loop (partwise), full loop)          done
# fuer alle modelle (all, sensor + motor, motor only)           done
# actinf velocity induced (all models) done
#TODO: pseudo random motorcommands for position prediction done

#TODO Daten vorbereiten:
# data:
#   - positions done
#   - sensor information    done (tensor)
#   - sensor predictions    done (tensor)
#   - position predictions of nearest obstacle (tensor)  done/noch aufaddieren
#       - 2 stage sensor induced action inference
#       - save optimized velocity (after actinf iterations)
#       + current thought position (cum sum of position delta pred.)
#   - acceleration  done
#   - velocity  done
#   - motor commands    done
#TODO
# train:
#  - static obstacles [[-1., 1.5], [1., 1.5], [-1.,0.5], [1.,0.5]], at pos. [0., 1.]
#       - plot losses
#  - no obstacles position [0., 1.], no border
#       - plot losses
#  - moving obstacles no movements
#       - actinf predict of sensor info -> position of obstacle
#       - plot distance from thought to reality position of obstacle
#  - moving obstacles agent moving
#       - actinf predict of sensor info -> position of obstacle
#       - plot distance from thought to reality position of obstacle

#TODO
# draw:
#   - draw each time step + enter time step (go back, go front)
#   - for each time step draw:
#       sensor information,
#       position of obstacle thought (10),
#       current velocity,
#       current motor commands,
#       current acceleration,
#       predicted sensor information (10),
