from tkinter import *

import numpy as np


def transform_pos(pos, min_x, max_x, min_y, max_y, new_width, new_height):
    x = pos
    old_width = max_x - min_x #3
    old_height = max_y - min_y #2
    new_x = [x[0] - min_x, np.abs(x[1]  - max_y)]
    new_x = [new_width / old_width * new_x[0], new_height/ old_height * new_x[1]]
    return new_x


def draw_path(positions, targets, obstacles):
    master = Tk()

    w = Canvas(master, width=600, height=400)
    w.pack()
    # draw border
    w.create_rectangle(0, 0, 600, 400, fill='white')
    for i in range(6):
        w.create_line(i * 100, 0, i*100, 400, fill='grey')
    for i in range(4):
        w.create_line(0, i*100, 600, i*100, fill='grey')

    # draw_obstacles
    for p in targets:
        p = transform_pos(p, -1.5, 1.5, 0, 2, 600, 400)
        r = 10
        w.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, fill='red')
        w.create_text(p[0], p[1], text='t')

    for p in obstacles:
        p = transform_pos(p, -1.5, 1.5, 0, 2, 600, 400)
        r = 10
        w.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, fill='yellow')
        w.create_text(p[0], p[1], text='o')

        # draw real positions
    for idx, pos in enumerate(positions):
        r = 10
        r1 = 2
        if idx == 0:
            pt = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            w.create_oval(pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r, fill='green')
            w.create_text(pt[0], pt[1], text='A')
        if idx + 1 < len(positions):
            pos = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            next_pos = positions[idx + 1]
            next_pos = transform_pos(next_pos, -1.5, 1.5, 0, 2, 600, 400)
            w.create_line(pos[0], pos[1], next_pos[0], next_pos[1], fill='cornflowerblue', width=3)
            if idx % 10 == 0:
                w.create_oval(pos[0] - r1, pos[1] - r1, pos[0] + r1, pos[1] + r1, fill='black')
        if idx == len(positions) - 1:
            pt = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
            w.create_oval(pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r, fill='green')
            w.create_text(pt[0], pt[1], text='A')
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
    mainloop()

def draw_timestep(sensor_data, positions, obstacles, timestep):
    master = Tk()
    w = Canvas(master, width=600, height=400)
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
    mainloop()
import matplotlib.pyplot as plt

def draw_sensor(sensor_data):
    bottom = 0
    max_height = 1
    size = 16

    width = (2 * np.pi) / size
    offset = width / 2.
    theta = np.linspace(offset, 2 * np.pi + offset, size, endpoint=False)
    radii = np.empty((size,))

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
    ax.set_ylim([0., 1.])
    sensor_data_bars = ax.bar(theta, radii, width=width, bottom=bottom, edgecolor='black', fill=False,
                                        linewidth=2)
    ax.set_title('sensor data')
    [sensor_data_bars[i].set_height(sensor_data[i]) for i in range(len(sensor_data_bars))]
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.show()

#draw_timestep([[[1 for i in range(16)]] for i in range(20)], [[0,1] for i in range(20)], [], 10)
#draw_sensor([0.5 for i in range(16)])
#TODO draw distence from agent to agent (Model A, B, C)
#
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import time

class LiveGui(Tk):
    def __init__(self, path, obstacles, target, sensor_pred_count=3, fast=False):
        super(LiveGui, self).__init__()
        self.target = target
        self.title('Actinf live View')
        self.minsize(640, 400)
        self.fps = 27
        self.button = Button(self,
                             text="PLAY/PAUSE", fg="black",
                             command=self.lulz)
        #self.button.pack(side=LEFT)
        self.button.grid(row= 1, column=3)

        self.sensor_pred_canvases = []
        self.button2 = Button(self,
                             text="Show thought obstacle position", fg="black",
                             command=self.lulz2)
        #self.button2.pack(side=LEFT)
        self.button2.grid(row= 2, column=3)
        self.fast = fast


        self.sensor_real = True
        self.pause = True
        self.sensor_pred_count = sensor_pred_count
        self.data = None
        self.load_data(path)
        self.draw_thought_obstacles = (len(self.data[5]) > 0)
        self.obstacles = obstacles
        self.tkinter_objects = []
        self.sensor_pred_plots = []
        self.time_step = 0
        self.max_time_step = len(self.data[0]) - 1
        self.scale = Scale(self, from_=0, to=self.max_time_step, length=600, tickinterval=10,orient=HORIZONTAL)
        self.scale2 = Scale(self, from_=1, to=100, length=600, tickinterval=10, orient=HORIZONTAL)
        #self.scale.pack(side=BOTTOM)
        self.scale.grid(row= 1, column=2)

        #self.scale2.pack(side=LEFT)
        self.scale2.grid(row= 2, column=2)
        self.scale2.set(self.fps)
        self.init_current_sensor_canvas()
        self.init_environment_canvas()
        if not self.fast:
            self.init_states_canvas()
            self.init_target_distance_canvas()
            self.init_obstacle_distance_canvas()
            self.init_sensor_pred_canvas()
        #self.init_sensor_pred_canvases()
        self.after(int(1000 / self.fps), self.start)
        self.mainloop()

    def init_states_canvas(self):
        states = self.data[7][self.time_step]

        f, ax = plt.subplots()
        ax.set_ylim([-1., 1.])
        self.current_hidden_plot = ax
        self.current_hidden_plot.plot(states[0][0][0])
        self.current_hidden_plot.set_title('States')
        self.current_hidden_canvas = FigureCanvasTkAgg(f, self)
        self.current_hidden_canvas.draw()
        #self.current_hidden_canvas.get_tk_widget().pack(side=LEFT, expand=False)
        self.current_hidden_canvas.get_tk_widget().grid(row=0, column=3)
        #self.current_hidden_canvas.grid(row=0, column=0)

    def update_state_canvas(self):
        self.current_hidden_plot.clear()
        self.current_hidden_plot.set_title('States')
        self.current_hidden_plot.set_ylim([-1., 1.])
        self.current_hidden_plot.plot(self.data[7][self.time_step][0][0][0], color='red')
        self.current_hidden_plot.plot(self.data[7][self.time_step][1][0][0], color='blue')
        self.current_hidden_canvas.draw()

    def init_sensor_pred_canvas(self):
        sensor_predictions = self.data[5][self.time_step]

        bottom = 0
        size = 16

        width = (2 * np.pi) / size
        offset = width / 2.
        theta = np.linspace(offset, 2 * np.pi + offset, size, endpoint=False)
        radii = np.empty((size,))
        plot_count = 3
        f, ax = plt.subplots(plot_count, subplot_kw=dict(polar=True))
        for idx, sensor_data in enumerate(sensor_predictions):
            if idx < plot_count:
                bar = ax[idx].bar(theta, radii, width=width, bottom=bottom, edgecolor='black',fill=False,linewidth=2)
                self.sensor_pred_plots.append(bar)
                [bar[i].set_height(sensor_data[i]) for i in range(len(bar))]

        self.sensor_pred_canvas = FigureCanvasTkAgg(f, self)
        self.sensor_pred_canvas.draw()
        self.sensor_pred_canvas.get_tk_widget().grid(row=1, column=4)

    def update_sensor_pred_canvas(self):
        sensor_predictions = self.data[5][self.time_step]
        for idx,bar in enumerate(self.sensor_pred_plots):
            [bar[i].set_height(sensor_predictions[idx][i]) for i in range(len(bar))]
        self.sensor_pred_canvas.draw()



    def lulz(self):
        if self.time_step == self.max_time_step:
            self.time_step = 0
            self.scale.set(0)
        else:
            self.pause = not self.pause

    def lulz2(self):
        self.draw_thought_obstacles = not self.draw_thought_obstacles

    def start(self):
        self.time_step = self.scale.get()
        self.fps = self.scale2.get()
        if self.time_step <= self.max_time_step:
            self.update_current_sensor_canvas()
            self.update_environment_canvas()
            if not self.fast:
                self.update_state_canvas()
                self.update_target_distance_canvas()
                self.update_obstacle_distance_canvas()
                self.update_sensor_pred_canvas()
            if not self.pause:
                self.scale.set(self.time_step + 1)
            self.after(int(1000/self.fps), self.start)
        else:
            self.time_step =0
            self.scale.set(0)

    def update_current_sensor_canvas(self):
        sensor_data = self.data[1][self.time_step][0]
        [self.current_sensor_data_bars[i].set_height(sensor_data[i]) for i in range(len(self.current_sensor_data_bars))]
        self.current_sensor_canvas.draw()

    def update_environment_canvas(self):
        position = self.data[0][self.time_step]
        position = transform_pos(position, -1.5, 1.5, 0, 2, 600, 400)
        position = [int(position[0]), int(position[1])]
        motor_commands = self.data[2][self.time_step]
        for tko in self.tkinter_objects:
            self.environment_canvas.delete(tko)
        self.tkinter_objects = []
        w = self.environment_canvas.create_rectangle(0,0,600,400, fill='white')
        for i in range(6):
            z = self.environment_canvas.create_line(i * 100, 0, i * 100, 400, fill='grey')
            self.tkinter_objects.append(z)
        for i in range(4):
            z = self.environment_canvas.create_line(0, i * 100, 600, i * 100, fill='grey')
            self.tkinter_objects.append(z)
        r = 10
        if self.draw_thought_obstacles and len(self.data[6]) > 0:
            self.draw_thought_obstacle_position(r)

        m1 = self.environment_canvas.create_line(position[0], position[1], #ru
                                                 position[0] + 15 * motor_commands[0],
                                                 position[1] + 15 * motor_commands[0], width=3)
        m2 = self.environment_canvas.create_line(position[0], position[1], #ro
                                                 position[0] + 15 * motor_commands[1],
                                                 position[1] - 15 * motor_commands[1],width=3)
        m3 = self.environment_canvas.create_line(position[0], position[1], #lu
                                                 position[0] - 15 * motor_commands[2],
                                                 position[1] + 15 * motor_commands[2], width=3)
        m4 = self.environment_canvas.create_line(position[0], position[1], \
                                                 position[0] - 15 * motor_commands[3],
                                                 position[1] - 15 * motor_commands[3], width=3)

        previous_positions = self.data[0][:self.time_step+1]
        for idx, pos in enumerate(previous_positions):
            if idx + 1 < len(previous_positions):
                pos = transform_pos(pos, -1.5, 1.5, 0, 2, 600, 400)
                next_pos = previous_positions[idx + 1]
                next_pos = transform_pos(next_pos, -1.5, 1.5, 0, 2, 600, 400)
                l = self.environment_canvas.create_line(pos[0], pos[1], next_pos[0], next_pos[1], fill='cornflowerblue', width=3)
                self.tkinter_objects.append(l)

        a = self.environment_canvas.create_oval(position[0] - r, position[1] - r, position[0] + r, position[1] + r,
                                                fill='yellow')
        self.tkinter_objects.append(a)
        self.tkinter_objects.append(m1)
        self.tkinter_objects.append(m2)
        self.tkinter_objects.append(m3)
        self.tkinter_objects.append(m4)
        self.tkinter_objects.append(w)
        # draw obstacles
        for o in self.obstacles:
            p = transform_pos(o, -1.5, 1.5, 0, 2, 600, 400)
            r = 10
            o = self.environment_canvas.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, fill='red')
            self.tkinter_objects.append(o)

        r = 10
        target = transform_pos(self.target, -1.5, 1.5, 0, 2, 600, 400)
        self.environment_canvas.create_oval(target[0] - r, target[1] - r, target[0] + r,
                                            target[1] + r, fill='green')


    def draw_thought_obstacle_position(self, r):
        obstacle_positions = self.data[0][self.time_step] + np.cumsum(self.data[6][self.time_step], axis=1)
        obstacle_positions = [transform_pos(v, -1.5, 1.5, 0, 2, 600, 400) for v in obstacle_positions]
        for p in obstacle_positions:
            r = 4
            o = self.environment_canvas.create_oval(int(p[0]) - r, int(p[1]) - r, int(p[0]) + r, int(p[1]) + r,
                                                    fill='blue')
            self.tkinter_objects.append(o)
        return r

    def init_target_distance_canvas(self):
        self.distances_to_position_target = [np.array(p) - np.array(self.target) for p in self.data[0]]
        self.distances_to_position_target = [np.sqrt(p[0]**2 + p[1]**2) for p in self.distances_to_position_target]
        f, ax = plt.subplots()
        ax.set_ylim([0, 3.])
        self.current_distance_plot = ax
        self.current_distance_plot.plot(self.distances_to_position_target[:self.time_step + 1])
        ax.set_title('Distance to target')
        self.current_distance_canvas = FigureCanvasTkAgg(f, self)
        self.current_distance_canvas.draw()
        # self.current_hidden_canvas.get_tk_widget().pack(side=LEFT, expand=False)
        self.current_distance_canvas.get_tk_widget().grid(row=0, column=4)

    def update_target_distance_canvas(self):
        self.current_distance_plot.clear()
        self.current_distance_plot.set_title('Distance to target')
        self.current_distance_plot.set_ylim([0., 3.])
        self.current_distance_plot.plot(self.distances_to_position_target[:self.time_step + 1], color='red')
        self.current_distance_canvas.draw()

    def get_closest_obstacle_distance(self, p):
        min = np.inf
        for o in self.obstacles:
            v = np.array(p) - np.array(o)
            d = np.sqrt(v[0]**2 + v[1]**2)
            if d < min:
                min = d
        return min

    def init_obstacle_distance_canvas(self):
        self.distances_to_nearest_obstacle = [self.get_closest_obstacle_distance(p) for p in self.data[0]]
        f, ax = plt.subplots()
        ax.set_ylim([0, 3.])
        self.current_obstacle_distance_plot = ax
        self.current_obstacle_distance_plot.plot(self.distances_to_nearest_obstacle[:self.time_step + 1])
        ax.set_title('Distance to nearest obstacle')
        self.current_obstacle_canvas = FigureCanvasTkAgg(f, self)
        self.current_obstacle_canvas.draw()
        # self.current_hidden_canvas.get_tk_widget().pack(side=LEFT, expand=False)
        self.current_obstacle_canvas.get_tk_widget().grid(row=1, column=1)

    def update_obstacle_distance_canvas(self):
        self.current_obstacle_distance_plot.clear()
        self.current_obstacle_distance_plot.set_title('Distance to nearest obstacle')
        self.current_obstacle_distance_plot.set_ylim([0., 3.])
        self.current_obstacle_distance_plot.plot(self.distances_to_nearest_obstacle[:self.time_step + 1], color='red')
        self.current_obstacle_canvas.draw()

    def init_environment_canvas(self):
        position = self.data[0][self.time_step]
        self.environment_canvas = Canvas(self, width=600, height=400)
        #self.environment_canvas.pack()
        self.environment_canvas.grid(row= 0, column=2)
        r = 10
        position = transform_pos(position, -1.5, 1.5, 0, 2, 600, 400)
        position = [int(position[0]), int(position[1])]
        a = self.environment_canvas.create_oval(position[0] - r, position[1] - r, position[0] + r, position[1] + r, fill='yellow')
        self.tkinter_objects.append(a)
        # draw obstacles
        for o in self.obstacles:
            p = transform_pos(o, -1.5, 1.5, 0, 2, 600, 400)
            r = 10
            o = self.environment_canvas.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, fill='red')
            self.tkinter_objects.append(o)

        r = 10
        target = transform_pos(self.target, -1.5, 1.5, 0, 2, 600, 400)
        self.environment_canvas.create_oval(target[0] - r, target[1] - r, target[0] + r,
                                            target[1] + r, fill='green')

    def init_current_sensor_canvas(self):
        sensor_data = self.data[1][self.time_step][0]
        bottom = 0
        size = 16

        width = (2 * np.pi) / size
        offset = width / 2.
        theta = np.linspace(offset, 2 * np.pi + offset, size, endpoint=False)
        radii = np.empty((size,))

        f, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
        ax.set_ylim([0., 2.])
        self.current_sensor_data_bars = ax.bar(theta, radii, width=width, bottom=bottom, edgecolor='black', fill=False,
                                  linewidth=2)
        ax.set_title('sensor data')
        [self.current_sensor_data_bars[i].set_height(sensor_data[i]) for i in range(len(self.current_sensor_data_bars))]
        self.current_sensor_canvas = FigureCanvasTkAgg(f, self)
        self.current_sensor_canvas.draw()
        #self.current_sensor_canvas.get_tk_widget().pack(side=RIGHT, expand=False)
        self.current_sensor_canvas.get_tk_widget().grid(row= 0, column=1)

    def load_data(self, path):
        self.data = np.load(path, allow_pickle=True)

    def update(self, obstacles):
        pos = self.data[self.time_step]['position']
        acc = self.data[self.time_step]['acceleration']
        sensor = self.data[self.time_step]['sensor']
        sensor_pred = None

        self.update_timestep(pos, acc, sensor, sensor_pred, obstacles)

    def change_current_time_step(self, time_step):
        if time_step < 0 or time_step > self.max_time_step:
            raise Exception('no valid time step, draw.py')
        else:
            self.time_step = time_step


if __name__ =='__main__':
    g = LiveGui('./testing_models/run_D.npy', [[-1., 1.5], [1., 1.5], [-1.,0.5], [1.,0.5]], [1., 1.8],\
                fast=False)

#Models:
#without border
# /models/with_border/moving_obstacles/run_A.npy done (okey)
# /models/with_border/moving_obstacles/run_A_seek.npy done (not bad)
# /models/with_border/moving_obstacles/run_B.npy done (kinda bad)
# /models/with_border/moving_obstacles/run_B_seek.npy done (bad)
# /models/with_border/moving_obstacles/run_C.npy done (good)
# /models/with_border/moving_obstacles/run_C_seek.npy done (bad)

#without border
# /models/with_border/no_obstacles/run_A.npy done (good)
# /models/with_border/no_obstacles/run_A_seek.npy done (bad)
# /models/with_border/no_obstacles/run_B.npy done (not that bad)
# /models/with_border/no_obstacles/run_B_seek.npy done (bad)
# /models/with_border/no_obstacles/run_C.npy done (ok)
# /models/with_border/no_obstacles/run_C_seek.npy done (bad)

#with border
# /models/with_border/static_obstacles/run_A.npy done (ok)
# /models/with_border/static_obstacles/run_A_seek.npy done (bad)
# /models/with_border/static_obstacles/run_B.npy done (bad)
# /models/with_border/static_obstacles/run_B_seek.npy done (bad)
# /models/with_border/static_obstacles/run_C.npy done (ok)
# /models/with_border/static_obstacles/run_C_seek.npy done (bad)