import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
from tkinter import *

import global_config as c
from simulator import Simulator
from data import Data
from gui import GUI
from agent import Agent
from stopwatch import Stopwatch
from plot import Plot

def start():

    # CONSTANTS
    NUM_EPOCHS = 200  # Amount of epochs of size NUM_MINI_EPOCHS
    NUM_MINI_EPOCHS = 20  # Amount of mini-epochs of size NUM_TIME_STEPS
    NUM_TIME_STEPS = 15
    LEARNING_RATE = 0.01

    gui = GUI()

    s = Stopwatch()
    sim = Simulator(mode=1, stopwatch=s)

    agents = []
    a1 = Agent(id='A', color='red', init_pos=np.array([0., 1.]), lr=LEARNING_RATE , gui=gui, show_sensor_plot=True, sim=sim, num_epochs=NUM_EPOCHS)
    agents.append(a1)
    
    # a2 = Agent(id='B', color='green', init_pos=np.array([0., 1.4]), lr=LEARNING_RATE , gui=gui, sim=sim, num_epochs=NUM_EPOCHS)
    # agents.append(a2)
   
    obstacles = []
    obstacles = create_obstacles(1, gui=gui, sim=sim, positions=np.array([[0., 1.]]))
    # obstacles = create_obstacles(1, gui=gui, sim=sim)

    for a in agents:
        a.register_agents(obstacles)

        if a.gui_att is not None:
            a.gui_att.show_target = False

    # a1.register_agents([a2])
    # a2.register_agents([a1])
    
    def callback(event):
        pos_gui = np.array([event.x, event.y])
        pos = gui.unscale(np.array([event.x, event.y]))
        print("clicked at", pos_gui, pos)

        a1.data.positions.write(pos[np.newaxis, :], 0)
        sensor_data = a1.sim.calc_sensor_data(from_step=0, to_step=1, agent=a1)[0]
        a1.gui_att.update_position(pos, np.zeros([4]), sensor_data)

        a1.sensorplot.update(sensor_data, np.zeros([c.INPUT_SENSOR_DIM]))


    gui.panel.bind("<Button-1>", callback)
    gui.panel.pack()
    
    while(True):
        gui.draw()


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

        o = Agent(id='O'+str(obstacle_index), color='gray', init_pos=position, lr=None , gui=gui, sim=sim, is_obstacle=True)

        o.data.positions.append_single(position)
        o.data.positions.change_curr_idx(-1)

        obstacles.append(o)

        if o.gui_att is not None:
            o.gui_att.show_target = False
            o.gui_att.show_simulated_positions = False

    return obstacles


start()