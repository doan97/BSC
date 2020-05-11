
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from scipy.spatial.qhull import ConvexHull

epochs = 50
mini_epochs = 30
time_steps = 15
all_time_steps = epochs * mini_epochs * time_steps
all_mini_epochs = epochs * mini_epochs
losses = np.load('only_sensor_loss.npy')
losses2 = np.load('marco_loss.npy')
d1 = np.load('distance_A.npy')
d2 = np.load('distance_B.npy')
h1 = np.load('all_hidden_states_A.npy', allow_pickle=True)
h2 = np.load('all_hidden_states_B.npy', allow_pickle=True)

one_h = np.load('one_time_step_hidden_states_A.npy', allow_pickle=True)
one_h2 = np.load('one_time_step_hidden_states_B.npy', allow_pickle=True)
one_c = np.load('one_time_step_cell_states_A.npy', allow_pickle=True)
one_c2 = np.load('one_time_step_cell_states_B.npy', allow_pickle=True)

ten_h1 = np.load('10_time_step_hidden_states_A.npy', allow_pickle=True)
ten_h2 = np.load('10_time_step_hidden_states_B.npy', allow_pickle=True)

num_hidden = 36
actinf_time_steps = 10
time_steps_prediction = 10
actinf_iterations = 10
target_change_frequency = 30
assert len(losses) == epochs * mini_epochs, print(len(losses))
def plot_net_loss(losses, flatten=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for epoch in range(epochs):
        data_for_epoch = []
        x = []
        y = []
        z = epoch
        for mini_epoch in range(mini_epochs):

            x.append(mini_epoch)
            y.append(sum(losses[mini_epoch + (epoch * mini_epochs)]))
        if flatten:
            ax.plot(x, flatten_data(y), z, zdir='y', c='blue')
        else:
            ax.plot(x,y,z, zdir='y', c = 'orange')
    plt.show()

def plot_net_loss_position(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for epoch in range(epochs):
        data_for_epoch = []
        x = []
        y = []
        z = epoch
        for mini_epoch in range(mini_epochs):

            x.append(mini_epoch)
            y.append(losses[mini_epoch + (epoch * mini_epochs)][0])
        ax.plot(x, y, z, zdir='y', c='blue')
    plt.show()

def plot_net_loss_sensor(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for epoch in range(epochs):
        data_for_epoch = []
        x = []
        y = []
        z = epoch
        for mini_epoch in range(mini_epochs):

            x.append(mini_epoch)
            y.append(losses[mini_epoch + (epoch * mini_epochs)][1])
        ax.plot(x, y, z, zdir='y', c='blue')
    plt.show()

def plot_net_loss_acceleation(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for epoch in range(epochs):
        data_for_epoch = []
        x = []
        y = []
        z = epoch
        for mini_epoch in range(mini_epochs):

            x.append(mini_epoch)
            y.append(losses[mini_epoch + (epoch * mini_epochs)][2])
        ax.plot(x, y, z, zdir='y', c='blue')
    plt.show()

def plot_simpler(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #prepare data
    losses = np.split(losses, mini_epochs)
    ax.plot_surface([i for i in range(mini_epochs)], losses, [i for i in range(epochs)], zdir='y', c='blue')

def compare_plot(loss1, loss2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')
    data = np.array([sum(items)/time_steps for items in loss1])
    data2 = np.array([sum(items)/time_steps for items in loss2])
    ax.plot([i for i in range(all_mini_epochs)],data, 0, zdir='y', label='sensor and motorcommand inputs')
    ax.plot([i for i in range(all_mini_epochs)],[0 for i in range(all_mini_epochs)], 0, zdir='y', c='red', linewidth=0.5)
    #ax.plot([i for i in range(all_mini_epochs)],data - data2, 1, zdir='y', label='difference')
    #ax.plot([i for i in range(all_mini_epochs)],[0 for i in range(all_mini_epochs)], 1, zdir='y',c='red')
    ax.plot([i for i in range(all_mini_epochs)],data2, 2, zdir='y', label='all inputs', c='green')
    ax.plot([i for i in range(all_mini_epochs)],[0 for i in range(all_mini_epochs)], 2, zdir='y',c='red', label='zero', linewidth=0.5)

    ax.set_yticklabels([])

    ax.set_xlabel('mini epochs')
    ax.set_zlabel('loss')
    #ax.legend()
    #plt.legend(loc='upper left', borderaxespad=0.)
    plt.show()

def compare_diffenrent_losses(loss):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')
    position_loss = np.array([items[0]/time_steps for items in loss])
    sensor_loss = np.array([items[1]/time_steps for items in loss])
    acceleration_loss = np.array([items[2]/time_steps for items in loss])
    ax.plot([i for i in range(all_mini_epochs)],position_loss, 0, zdir='y', label='sensor and motorcommand inputs', c='blue')
    ax.plot([i for i in range(all_mini_epochs)],[0 for i in range(all_mini_epochs)], 0, zdir='y', c='red', linewidth=0.5)
    ax.plot([i for i in range(all_mini_epochs)],sensor_loss, 1, zdir='y', label='all inputs', c='black')
    ax.plot([i for i in range(all_mini_epochs)],[0 for i in range(all_mini_epochs)], 1, zdir='y',c='red', label='zero', linewidth=0.5)
    ax.plot([i for i in range(all_mini_epochs)], acceleration_loss, 2, zdir='y', label='all inputs', c='green')
    ax.plot([i for i in range(all_mini_epochs)], [0 for i in range(all_mini_epochs)], 2, zdir='y', c='red',label='zero', linewidth=0.5)

    ax.set_yticklabels([])

    ax.set_xlabel('mini epochs')
    ax.set_zlabel('loss')
    #ax.legend()
    #plt.legend(loc='upper left', borderaxespad=0.)
    plt.show()

def plot_distances(d1, d2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.gca(projection='3d')
    ax.plot([i for i in range(actinf_time_steps)], d1, label='sensor and motorcommand inputs',
            c='green')
    ax.plot([i for i in range(actinf_time_steps)], d2, c='blue')
    ax.plot([0, actinf_time_steps],[sum(d1)/actinf_time_steps,sum(d1)/actinf_time_steps], c='red')
    ax.plot([0, actinf_time_steps],[sum(d2)/actinf_time_steps,sum(d2)/actinf_time_steps], c='orange')
    a = 3
    b = 2
    max_distance = np.sqrt(a**2 + b**2)
    for i in range(int(actinf_time_steps/ target_change_frequency)):

        ax.plot([i * target_change_frequency, i*target_change_frequency], [0, max_distance], "k--")

    #ax.set_yticklabels([])

    ax.set_ylabel('distance to target')
    ax.set_xlabel('time steps')
    # ax.legend()
    # plt.legend(loc='upper left', borderaxespad=0.)
    plt.show()

def flatten_data(data, mean_neighbors= 30):
    if len(data) < mean_neighbors:
        print('nooo')
        return
    if mean_neighbors % 2 == 1:
        left_neighbors = int((mean_neighbors-1)/2)
        right_neighbors = int((mean_neighbors-1)/2)
    else:
        left_neighbors = int((mean_neighbors-1)/2)
        right_neighbors = int((mean_neighbors-1)/2) + 1

    for idx, item in enumerate(data):
        left_value = 0
        right_value = 0
        missing_neighbors_left = left_neighbors - idx
        missing_neighbors_right = idx - (len(data) - right_neighbors)
        self_value = item
        if missing_neighbors_left > 0:
            self_value += missing_neighbors_left * item
        if missing_neighbors_right > 0:
            self_value += missing_neighbors_right * item

        if idx < left_neighbors or idx > (len(data) - right_neighbors):
            if idx < left_neighbors:
                for i in range(left_neighbors - missing_neighbors_left):
                    left_value += data[idx - (i + 1)]
            if idx > (len(data) - right_neighbors):
                for i in range(right_neighbors - missing_neighbors_right):
                    right_value += data[idx - (i + 1)]
        else:
            for i in range(left_neighbors):
                left_value += data[idx - (i + 1)]
            for i in range(right_neighbors):
                right_value += data[idx - (i + 1)]

        data[idx] = (left_value + self_value + right_value) / mean_neighbors

    return data

def convex_hull(data):
    ydata = data
    xdata = [i for i in range(len(ydata))]
    points = np.vstack((xdata, ydata)).T

    hull = ConvexHull(points)

    plt.plot(points[:, 0], points[:, 1], 'o')

    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.show()

def get_vector_length(v):
    return np.sqrt(v[0]**2 + v[1]**2)

def plot_loss_one_axis(loss):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i for i in range(len(loss))], [sum(item) for item in loss])
    plt.show()

def plot_loss_per_epoch(loss, mode=2):
    epoch_values = []
    for i in range(len(loss)):
        if ((i+1) % mini_epochs == 0) and not i == 0:
            print(i)
            if mode == 0:
                epoch_values.append(sum(loss[i]))
            if mode == 1:
                epoch_values.append(loss[i][0])
            if mode == 2:
                epoch_values.append(loss[i][1])
            if mode == 3:
                epoch_values.append(loss[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i for i in range(epochs)], epoch_values)
    plt.show()

def plot_hidden_states_3d(data):
    data_h0 = [d[0][0] for d in data]

    hs = []
    for i in range(36):
        h = []
        for j in range(len(data_h0)):
            h.append(data_h0[j][i])
        hs.append(h)

    x = [i for i in range(len(data))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(36):
        ax.plot(x, hs[i], i, zdir='y', linewidth=0.4)
    ax.set_xlabel('time step')
    ax.set_ylabel('hidden state')
    ax.set_zlabel('hidden value')

    ax.set_yticklabels([])
    plt.show()

def plot_cell_states_3d(data):
    data_c0 = [d[1][0] for d in data]

    hs = []
    for i in range(36):
        h = []
        for j in range(len(data_c0)):
            h.append(data_c0[j][i])
        hs.append(h)
    x = [i for i in range(len(data))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_hidden):
        ax.plot(x, hs[i], i, zdir='y', linewidth=0.4)
    ax.set_xlabel('time step')
    ax.set_ylabel('cell state')
    ax.set_zlabel('cell value')

    ax.set_yticklabels([])
    plt.show()

def plot_one_time_step_hidden_states(all_h2):
    #print(all_h)
    #print(type(all_h[0]))
    all_h = all_h2[1:]
    #print(all_h)
    #print(all_h[1].detach().numpy())
    for idx, h in enumerate(all_h):
        if isinstance(h, torch.Tensor):
            h = h
        elif isinstance(h, float):
            h = torch.tensor(np.zeros(36))
        #print(h)
        all_h[idx] = h.detach().numpy()
    #(all_h)
    #all_h = [h.detach().numpy() for h in all_h]
    hs = []
    for i in range(36):
        h = []
        for j in range(len(all_h)):
            h.append(all_h[j][i])
        hs.append(h)
    x = [i for i in range(len(all_h))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    for i in range(num_hidden):
        #ax.plot(x, hs[i], i, zdir='y', linewidth=0.4)
        ax.plot(x, hs[i], linewidth=0.8)
    ax.set_xlabel('predicted time steps')
    ax.set_ylabel('hidden value')

    for i in range(actinf_time_steps):
        print(i * (time_steps_prediction * actinf_iterations))
        ax.plot([i * (time_steps_prediction * actinf_iterations), i*(actinf_time_steps * actinf_iterations)], [-0.6, 0.3], "k--")
    #ax.set_zlabel('cell value')
    plt.show()

class ActinfPlotter:
    def __init__(self, datas, runs, timesteps ,actinf_iterations, actinf_prediction_horizon):
        self.datas = datas
        self.runs = runs
        self.timesteps = timesteps
        self.actinf_iterations = actinf_iterations
        self.actinf_prediction_horizon = actinf_prediction_horizon
        self.num_hidden = 36

        self.needed_sequences = {'one step':[0,1], \
                                 'ten steps':[0,10],\
                                 'all steps': [0, len(self.datas)]}

        if self.check_data():
            self.data_to_plot = self.prepare_data()
            #self.plot_data_average()
            #self.plot_hidden(sequence='one step')
            self.make_all_plots_one_run(sequence='all steps')
        else:
            print('shapes does not match')
            return

    def make_all_plots_one_run(self, run=0, sequence=None):
        #self.plot_data_average()
        self.plot_hidden(run=run, sequence=sequence, plot='hidden')
        self.plot_hidden(run=run, sequence=sequence, plot='cell')
        data = self.data_to_plot[run][sequence]
        self.plot_normal(data)

    def plot_normal(self, data):
        coms = data['coms']

        perf = data['perf']
        vel = data['vel']
        vel = [np.linalg.norm(v) for v in vel]
        loss = data['loss']
        self.plot_linear(vel, 'velocity')
        self.plot_linear(perf, 'distance to target')
        self.plot_linear(loss, 'loss')
        #plot perf

    def plot_linear(self, data, label=''):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(num_hidden):
            # ax.plot(x, hs[i], i, zdir='y', linewidth=0.4)
            ax.plot([i for i in range(len(data))], data, linewidth=0.8)
        ax.set_xlabel('time steps')
        ax.set_ylabel(label)
        plt.show()

    def make_all_average_plot(self, com=False, vel=True, perf=True, loss=False):
        pass

    def plot_hidden(self, run=0, sequence=None, plot='hidden'):
        if sequence is None:
            return
        all_h = self.data_to_plot[run][sequence]['hidden']
        if plot == 'hidden':
            all_h = [h[0] for h in all_h]
        elif plot == 'cell':
            all_h = [h[1] for h in all_h]

        for idx, h in enumerate(all_h):
            if isinstance(h, torch.Tensor):
                h = h.detach().numpy()[0][0]
            elif isinstance(h, float):
                h = np.zeros(self.num_hidden)
            elif isinstance(h, np.ndarray):
                h = h[0]
            all_h[idx] = h

        hs = []
        for i in range(self.num_hidden):
            h = []
            for j in range(len(all_h)):
                h.append(all_h[j][i])
            hs.append(h)
        x = [i for i in range(len(all_h))]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(self.num_hidden): #len(hs) : num of hidden neurons
            ax.plot(x, hs[i], linewidth=0.8)
        ax.set_xlabel('predicted time steps')
        ax.set_ylabel('hidden value')
        plt.show()

    def plot_data_average(self):
        """
        for each sequence needed plot average of all runs:
            - coms, perf, vel, loss, hidden

        self.data_to_plot, array
            run in self.data_to_plot dict : (seq_names)
        """
        averages = {}
        for seq_name in self.needed_sequences:
            averages[seq_name] = None


        for run in self.data_to_plot:
            for sequence_name in run:
                data = run[sequence_name]

                if averages[sequence_name] is not None:
                    averages[sequence_name] += data
                else:
                    averages[sequence_name] = data

        for key in averages:
            for data_type in averages[key]: #commands, vel,...
                if not data_type == 'hidden':
                    averages[key][data_type] /= self.runs
            #verages[key] /= self.runs

        assert averages['ten steps'] == self.data_to_plot[0]['ten steps']





    def check_data(self):
        """
        every_data_contains:
        motor commands : (timesteps, 4) array
        hidden states : (timesteps, 2, 36) h and c Tensor
        performances : (timesteps,) float
        loss to target velocity : (timesteps,) float
        """

        assert len(self.datas) == self.runs
        for run in self.datas:
            assert len(run) == 5 #coms, hidden, vel, perf, loss
            assert len(run['coms']) == self.timesteps
            assert len(run['perf']) == self.timesteps
            assert len(run['vel']) == self.timesteps
            assert len(run['loss']) == self.timesteps * self.actinf_iterations
            assert len(run['hidden']) == \
                   self.timesteps * self.actinf_iterations * self.actinf_prediction_horizon

        return True

    def prepare_data(self):
        runs = []
        for run in self.datas:
            data_to_plot = {}
            for  key in self.needed_sequences:
                sequence = self.needed_sequences[key]
                data = {'coms': np.array(run['coms'][sequence[0]: sequence[1]]),\
                        'perf': np.array(run['perf'][sequence[0]: sequence[1]]), \
                        'vel': np.array(run['vel'][sequence[0]: sequence[1]]), \
                        'loss': np.array(run['loss'][sequence[0]: sequence[1]]),\
                        'hidden': run['hidden'][sequence[0] * self.actinf_iterations * self.actinf_prediction_horizon: \
                                                sequence[1] * self.actinf_iterations * self.actinf_prediction_horizon]}
                t = run['hidden'][sequence[0] * self.actinf_iterations * self.actinf_prediction_horizon: \
                                                sequence[1] * self.actinf_iterations * self.actinf_prediction_horizon]

                for idx,i in enumerate(t):
                    t[idx] = [z[0] for z in i]

                data_to_plot[key] = data
            runs.append(data_to_plot)
        return runs




#one = torch.load('datas/one_time_step_datas_A')
#ten = torch.load('datas/ten_time_step_datas_A')
#all = torch.load('datas/all_time_step_datas_A')
d = torch.load('all_runs_A')
p = ActinfPlotter(d, 2, 200, 10, 10)

#check_data()

#plot_one_time_step_hidden_states(ten_h1)
#plot_one_time_step_hidden_states(ten_h2)
#plot_one_time_step_hidden_states(one_h)
#plot_one_time_step_hidden_states(one_h2)
#plot_one_time_step_hidden_states(one_c)
#plot_one_time_step_hidden_states(one_c2)
#plot_hidden_states_3d(h)
#plot_hidden_states_3d(h2)
#plot_cell_states_3d(h1)
#plot_cell_states_3d(h2)
#plot_loss_per_epoch(losses)
#plot_loss_one_axis(losses)
#convex_hull([sum(item) for item in losses])
#plot_net_loss(losses)
#plot_net_loss_position(losses)
#plot_net_loss_sensor(losses)
#plot_net_loss_acceleation(losses)
#plot_simpler(losses)
#compare_plot(losses, losses2)
#compare_diffenrent_losses(losses)
#compare_diffenrent_losses(losses2)
#plot_distances(d1, d2)

