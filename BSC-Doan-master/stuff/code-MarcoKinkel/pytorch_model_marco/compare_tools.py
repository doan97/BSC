import os
import numpy as np
from pathlib import Path
import torch
from PyQt5.QtCore import pyqtSlot

from PyQt5.QtWidgets import QPushButton, QSizePolicy, QTreeView, QFileSystemModel, QLabel, QProgressBar, QCheckBox
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


class NavigationToolbar(object):
    pass


class Tools(QtWidgets.QMainWindow):

    def __init__(self,obj):
        self.obj = obj
        self.obj._main = QtWidgets.QWidget()
        self.obj.setCentralWidget(self.obj._main)
        self.obj.grid = QtWidgets.QGridLayout(self.obj._main)

        self.data1 = []
        self.data2 = []

        self.models = []
        self.one_step = True
        self.two_step = True
        self.time_steps = 10
        self.canvases = []

    def train(self, name, v_noise=False, v_drop=False, val_noise=None, val_drop=None):
        NUM_EPOCHS = 200  # Amount of epochs of size NUM_MINI_EPOCHS
        NUM_MINI_EPOCHS = 30  # Amount of mini-epochs of size NUM_TIME_STEPS
        NUM_TIME_STEPS = 15
        LEARNING_RATE = 0.01
        model_name = name

        if not v_noise and not v_drop:
            os.system('python3 ./compare_train.py ' \
                      + str(NUM_EPOCHS) + ' '
                      + str(NUM_MINI_EPOCHS) + ' '
                      + str(NUM_TIME_STEPS) + ' '
                      + str(LEARNING_RATE) + ' '
                      + model_name)
        else:
            os.system('python3 ./compare_train.py ' \
                      + str(NUM_EPOCHS) + ' '
                      + str(NUM_MINI_EPOCHS) + ' '
                      + str(NUM_TIME_STEPS) + ' '
                      + str(LEARNING_RATE) + ' '
                      + model_name + ' '
                      + 'True' + ' ' #velocity noise
                      + 'False' + ' ' #velocity dropout
                      + str(val_noise) + ' '
                      + str(val_drop))

    def make_checkboxes_and_progress(self):
        self.obj.checkbox_one_step = QCheckBox("One-step", self.obj)
        self.obj.checkbox_one_step.stateChanged.connect(self.check_one)
        self.obj.checkbox_two_step = QCheckBox("two-step", self.obj)
        self.obj.checkbox_two_step.stateChanged.connect(self.check_two)
        self.obj.checkbox_one_step.setChecked(True)
        self.obj.checkbox_two_step.setChecked(True)

        progress = QProgressBar(self.obj)
        progress.setGeometry(200, 80, 250, 20)

        sublayout = QtWidgets.QBoxLayout(0, parent=self.obj._main)
        sublayout.addWidget(progress)
        sublayout.addWidget(self.obj.checkbox_one_step)
        sublayout.addWidget(self.obj.checkbox_two_step)

        self.obj.grid.addLayout(sublayout, 0, 1)

    def check_one(self):
        self.one_step = not self.one_step
        print('onestep', self.one_step)

    def check_two(self):
        self.two_step = not self.two_step
        print('twostep', self.two_step)

    def train_multiple_noise(self):
        model_name = 'v_noise_'

        for i in range(6):
            noise_rate = (i+1) * 0.1
            name = model_name + str(noise_rate)

            self.train(name, v_noise=True, v_drop=False, val_noise=noise_rate, val_drop=0)

    def train_multiple_dropout(self):
        model_name = 'v_drop_'

        for i in range(6):
            dropout_rate = (i+1) / 10
            name = model_name + str(dropout_rate)

            self.train(name, v_noise=False, v_drop=True, val_noise=0, val_drop=dropout_rate)

    def train_multiple_nd(self):
        model_name = 'v_'
        for i in range(6):
            noise_rate = (i + 1) * 0.1
            name = model_name + str(noise_rate) + '_'
            for j in range(6):
                dropout_rate = (j + 1) / 10
                c_name = name + str(dropout_rate)
                self.train(c_name, v_noise=True, v_drop=True, val_noise=noise_rate, val_drop=dropout_rate)




    def run_actinf(self):
        self.models = self.get_folder_inhalt('compare_models/')
        a = 0
        b = 0
        if self.one_step: a = 1; self.reset_folder('datas/')
        if self.two_step: b = 1; self.reset_folder('datas2/')
        full_percent = (a + b) * len(self.models)
        counter = 0


        for model_name in self.models:
            id = self.models.index(model_name)
            if self.one_step:
                os.system(
                    'python3 ./temp_actinf.py ./compare_models/' + model_name + ' ' + str(id) + ' ' + '1step' + ' ' + str(self.time_steps))
                counter += 1
                self.obj.progress.setValue(int(counter / full_percent * 100))
            if self.two_step:
                os.system(
                    'python3 ./temp_actinf.py ./compare_models/' + model_name + ' ' + str(id) + ' ' + '2step' + ' ' + str(self.time_steps))
                counter += 1
                self.obj.progress.setValue(int(counter / full_percent * 100))

    def reset_folder(self, folder):
        os.system('rm ./' + folder + '* 2> /dev/null')

    def get_folder_inhalt(self, folder):
        data = []
        os.system('ls ./' + folder + ' > folder_inhalt')
        with open('folder_inhalt', 'r') as infile:
            for line in infile:
                data.append(line[:len(line) - 1])
        infile.close()

        return data


    def calc_average(self):
        datas = self.load_data()
        avrg1 = []
        avrg2 = []

        for d1 in datas[0]:
            avrg1.append(sum(d1)/len(d1))
        for d2 in datas[1]:
            avrg2.append(sum(d2)/len(d2))

        return [avrg1, avrg2]



    def load_data(self, onestep=True, twostep=True):
        datas1 = self.get_folder_inhalt('datas/')
        datas2 = self.get_folder_inhalt('datas2/')
        data1 = []
        data2 = []
        if onestep:
            for d in datas1:
                d1 = np.load('./datas/' + d)
                data1.append(d1)
        if twostep:
            for d in datas2:
                d2 = np.load('./datas2/' + d)
                data2.append(d2)

        return [data1, data2]


    def plot_data(self):
        datas = self.load_data()
        avrg_data = self.calc_average()
        self.canv7.plot_avrg(avrg_data[0])
        self.canv8.plot_avrg(avrg_data[1])
        self.canv9.plot_in_one_canv(datas[0], 1)
        self.canv10.plot_in_one_canv(datas[1], 2)

    def dynamic_plot_data(self, onestep=True, twostep=True):
        datas = self.load_data()
        data1 = datas[0]
        data2 = datas[1]
        self.dynamic_plot_canvas(max(len(data1), len(data2)))
        c = 0
        for canv in self.canvases:
            if c < len(data1) or c < len(data2):
                if len(data1[c]) > 0 and len(data2[c]):
                    canv.plot(data1[c], data2[c], c)
                elif len(data1[c]) > 0:
                    canv.plot(data1[c], [])
                elif len(data2[c]) > 0:
                    canv.plot([], data2[c])
            else:
                return
            c += 1



    def get_perfomace_data_list(self):
        model_string = ''
        os.system('ls ./datas/ > data1')
        os.system('ls ./datas2/ > data2')
        path = Path('./')
        with open('data1', 'r') as infile:
            lines1 = [path + line.strip() for line in infile]
        infile.close()

        with open('data2', 'r') as infile:
            lines2 = [path + line.strip() for line in infile]
        infile.close()

        return [lines1, lines2]

    def create_plot_canvas(self):
        self.canv7 = PlotCanvas()
        self.canv8 = PlotCanvas()
        self.canv9 = PlotCanvas()
        self.canv10 = PlotCanvas()

        self.obj.grid.addWidget(self.canv7, 1, 4)
        self.obj.grid.addWidget(self.canv8, 1, 5)
        self.obj.grid.addWidget(self.canv9, 2, 4)
        self.obj.grid.addWidget(self.canv10, 2, 5)

    def dynamic_plot_canvas(self, counter):
        max_per_row = 3
        col = 0
        row = 1
        for i in range(counter):
            if col == max_per_row:
                col = 0
                row += 1
            canv = PlotCanvas()
            self.canvases.append(canv)
            self.obj.grid.addWidget(canv, row, col)
            col += 1




    def create_buttons_and_trees(self):
        self.b_add_mod = QPushButton('Add. model', self.obj)
        self.b_del_mod = QPushButton('Del. model', self.obj)

        self.b_mtrain_n = QPushButton('Multiple train noise', self.obj)
        self.b_mtrain_d = QPushButton('Multiple train drop', self.obj)
        self.b_mtrain_nd = QPushButton('Multiple train noise and drop', self.obj)

        self.b_train = QPushButton('Train', self.obj)
        self.b_plot = QPushButton('Plot', self.obj)
        self.b_test = QPushButton('Test', self.obj)
        self.b_test_load_plot = QPushButton('Test, Load, Plot', self.obj)
        self.b_test_load_plot.setStyleSheet("background-color: green")

        self.b_del_mods = QPushButton('Del. all models', self.obj)
        self.b_del_perf_1 = QPushButton('Del. 1-step perf.', self.obj)
        self.b_del_perf_2 = QPushButton('Del. 2-step perf.', self.obj)

        self.l_perf1 = QLabel('Perfomance (1):', self.obj)
        self.l_perf2 = QLabel('Perfomance (2):', self.obj)
        self.l_models = QLabel('Models :', self.obj)

        model = QFileSystemModel()
        model.setRootPath('./datas/')
        tree = QTreeView()
        tree.setModel(model)
        tree.setRootIndex(model.index('/home/v/BSC/BSC-Doan-master/stuff/code-MarcoKinkel/pytorch_model_marco/compare_models/'))

        tree.setAnimated(False)
        tree.setIndentation(20)
        tree.setSortingEnabled(True)
        tree.setWindowTitle("Performance(1)")



        tree2 = QTreeView()
        tree2.setModel(model)
        tree2.setRootIndex(model.index('/home/v/BSC/BSC-Doan-master/stuff/code-MarcoKinkel/pytorch_model_marco/datas/'))


        tree2.setAnimated(False)
        tree2.setIndentation(20)
        tree2.setSortingEnabled(True)
        tree2.setWindowTitle("Performance(2)")

        tree3 = QTreeView()
        tree3.setModel(model)
        tree3.setRootIndex(model.index('/home/v/BSC/BSC-Doan-master/stuff/code-MarcoKinkel/pytorch_model_marco/datas2/'))

        tree3.setAnimated(False)
        tree3.setIndentation(20)
        tree3.setSortingEnabled(True)
        tree3.setWindowTitle("Models")


        sublayout = QtWidgets.QBoxLayout(2,parent=self.obj._main)
        sublayout.addWidget(self.b_add_mod)
        sublayout.addWidget(self.b_del_mod)
        sublayout.addWidget(self.b_mtrain_n)
        sublayout.addWidget(self.b_mtrain_d)
        sublayout.addWidget(self.b_mtrain_nd)
        sublayout.addWidget(self.b_del_mods)
        sublayout.addWidget(self.b_del_perf_1)
        sublayout.addWidget(self.b_del_perf_2)

        sublayout2 = QtWidgets.QBoxLayout(2, parent=self.obj._main)
        sublayout2.addWidget(self.l_models)
        sublayout2.addWidget(tree)

        sublayout3 = QtWidgets.QBoxLayout(2, parent=self.obj._main)
        sublayout3.addWidget(self.l_perf1)
        sublayout3.addWidget(tree2)

        sublayout4 = QtWidgets.QBoxLayout(2, parent=self.obj._main)
        sublayout4.addWidget(self.l_perf2)
        sublayout4.addWidget(tree3)

        test_box = QtWidgets.QBoxLayout(0, self.obj._main)
        test_box.addWidget(self.b_test)
        test_box.addWidget(self.b_train)
        test_box.addWidget(self.b_plot)
        sublayout.addLayout(test_box)
        sublayout.addWidget(self.b_test_load_plot)

        self.obj.grid.addLayout(sublayout,0,2)
        self.obj.grid.addLayout(sublayout2,0,0)
        self.obj.grid.addLayout(sublayout3,0,4)
        self.obj.grid.addLayout(sublayout4,0,5)


    def give_buttons_onclick(self):
        self.b_test.clicked.connect(self.on_click_run)

    def create_fix_motorcommands(self, x):
        commands = [x**2, 3*x, -x**2, x/3] #sw, se, nw, ne
        maximum = max(commands)
        return [com/maximum for com in commands]

    def load_motorcommands(self):
        data = torch.load('commands.pt')
        return data




class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.used = False
        self.data = []
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def plot(self, data, data2, id):
        if self.used:
            self.figure.clear()
        self.used = True


        ax = self.figure.add_subplot(111)
        ax.set_xlabel('time steps')
        ax.set_ylabel('distance from A to B')
        ax.grid(True)
        if len(data) > 0:
            line1 = ax.plot(data, 'r-', label='one step')
        if len(data2) > 0:
            line2 = ax.plot(data2, 'g-', label='two step')
        else:
            ax.set_title('Not used')

        self.draw()

    def plot_in_one_canv(self, datas, step):
        if self.used:
            self.figure.clear()
        self.used = True


        ax = self.figure.add_subplot(111)
        ax.set_xlabel('time steps')
        ax.set_ylabel('distance from A to B')
        ax.grid(True)

        colors = ['r', 'g', 'b', 'c', 'y', 'k']
        ax.set_title('Performance compare '+ str(step) +'-step')
        c = 0
        for vals in datas:
            if len(vals) > 0:
                ax.plot(vals, colors[c], label=str(c))
                c += 1
        self.draw()

    def plot_avrg(self, data):
        if self.used:
            self.figure.clear()
        self.used = True


        ax = self.figure.add_subplot(111)
        ax.set_xlabel('time steps')
        ax.set_ylabel('distance from A to B')


        colors = ['r', 'g', 'b', 'c', 'y', 'k']
        y1 = data
        x1 = [i for i in range(len(data))]
        ax.bar(x1, y1, 1, color=colors)

        self.draw()