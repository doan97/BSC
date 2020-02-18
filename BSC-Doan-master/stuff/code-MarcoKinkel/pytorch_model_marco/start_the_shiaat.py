#!/usr/bin/python3.6
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.patches as mpatches

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QCheckBox, QLineEdit, QFileDialog, QLabel, QFrame, QProgressBar, QFileSystemModel, QTreeView, \
    QGridLayout
from PyQt5 import QtCore


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSlot
import json

class Start_compare():
    def __init__(self):
        app = QApplication(sys.argv)
        self.app = App()
        sys.exit(app.exec_())


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.models = []
        self.left = 10
        self.top = 10
        self.title = 'Vinhdo Doan Comparison of LSTMs'
        self.width = 1900
        self.height = 800
        self.initUI()




    def initUI(self):
        self.plot_one_step = True
        self.plot_two_step = True

        self.data1_items = 0
        self.data2_items = 0
        self.datas1 = [[] for i in range(6)]
        self.datas2 = [[] for i in range(6)]

        self.models_to_test = []
        self.canvas_positions = []


        self.canvas_width = 500
        self.canvas_height = 400

        self.p_width = self.canvas_width / 100
        self.p_height = self.canvas_height / 100

        self.x_rest_position = 3 * self.canvas_width + 30

        self.model = QFileSystemModel()
        self.model.setRootPath('./')
        self.tree = QTreeView()
        self.tree.setModel(self.model)

        self.tree.setAnimated(False)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)

        self.tree.setWindowTitle("Dir View")
        self.tree.resize(640, 480)
        self.tree.move(1520,0)

        self.m1 = PlotCanvas(self, width=5, height=4)
        self.m1.move(0, 0)
        self.m2 = PlotCanvas(self, width=5, height=4)
        self.m2.move(0 , self.canvas_height)

        self.m3 = PlotCanvas(self, width=5, height=4)
        self.m3.move(self.canvas_width, 0)
        self.m4 = PlotCanvas(self, width=5, height=4)
        self.m4.move(self.canvas_width, self.canvas_height)

        self.m5 = PlotCanvas(self, width=5, height=4)
        self.m5.move(2*self.canvas_width, 0)
        self.m6 = PlotCanvas(self, width=5, height=4)
        self.m6.move(2*self.canvas_width, self.canvas_height)

        self.setWindowTitle(self.title)

        self.count_canvas = 0

        self.create_rest()
        self.move_rest()
        self.update_textfields()
        self.resize_buttons()

        self.show()

    @pyqtSlot()
    def update_textfields(self):
        self.update_text(1)
        self.update_text(2)
        self.update_text(3)

    def make_string(self, liste):
        string = ''
        for it in liste:
            string += it + '\n'

        return string

    def create_rest(self):
        #run actinf and plot performance

        self.label1 = QLabel('',self)
        self.label1.move(1520, 330)
        self.label1.resize(310, 130)
        self.label1.setFrameShape(QFrame.Panel)
        self.label1.setLineWidth(1)
        self.time_steps = 10


        self.label2 = QLabel('', self)
        self.label2.move(1520, 650)
        self.label2.resize(310, 130)
        self.label2.setFrameShape(QFrame.Panel)
        self.label2.setLineWidth(1)

        self.label3 = QLabel('MODELLE:', self)
        self.label3.move(1520, 305)
        self.label3.resize(110, 20)

        self.label4 = QLabel('PERFORMANCE(2):', self)
        self.label4.move(1520, 625)
        self.label4.resize(130, 20)

        self.label5 = QLabel('PERFORMANCE(1):', self)
        self.label5.move(1520, 465)
        self.label5.resize(130, 20)

        self.label6 = QLabel('', self)
        self.label6.move(1520, 490)
        self.label6.resize(310, 130)
        self.label6.setFrameShape(QFrame.Panel)
        self.label6.setLineWidth(1)

        self.label7 = QLineEdit(str(self.time_steps), self)
        self.label7.move(1750, 270)
        self.label7.resize(50, 20)

        self.label8 = QLabel('TIME STEPS: current = ' + str(self.time_steps), self)
        self.label8.move(1520, 270)
        self.label8.resize(200, 20)

        self.label9 = QLabel('Testing', self)
        self.label9.move(1520, 240)
        self.label9.resize(70, 20)

        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        self.progress.move(1600, 240)

        self.create_buttons()
        self.create_checkbox()

    def play_label(self, counter):
        c = counter % 3 + 1
        if c == 1:
            self.label9.setText('Testing.')
        if c == 2:
            self.label9.setText('Testing..')
        if c == 3:
            self.label9.setText('Testing...')


    def create_buttons(self):
        self.button = QPushButton('RUN', self)
        self.button.clicked.connect(self.on_click_run)
        self.button.setToolTip('Start')
        #File Browse Button
        self.button2 = QPushButton('ADD MODEL', self)
        self.button2.clicked.connect(self.on_click_file_browse)
        self.button2.setToolTip('Start')

        self.button3 = QPushButton('LOAD DATA', self)
        self.button3.clicked.connect(self.on_click_file_load)
        self.button3.setToolTip('Start')

        self.button4 = QPushButton('PLOT', self)
        self.button4.clicked.connect(self.on_click_plot)
        self.button4.setToolTip('Start')

        self.button5 = QPushButton('DO NEW', self)
        self.button5.clicked.connect(self.get_models_out_of_folder)
        self.button5.setToolTip('Start')
        self.button5.setStyleSheet("background-color: green")

        self.button6 = QPushButton('PLOT PERFORMANCE', self)
        self.button6.clicked.connect(self.plot_from_folder)
        self.button6.setToolTip('Start')

        self.button8 = QPushButton('DEL', self)
        self.button8.clicked.connect(self.delete_mod_folder)
        self.button8.setToolTip('Start')
        self.button8.setStyleSheet("background-color: red")

        self.button7 = QPushButton('DEL', self)
        self.button7.clicked.connect(self.delete_perf_folder)
        self.button7.setToolTip('Start')
        self.button7.setStyleSheet("background-color: red")

        self.button9 = QPushButton('MOD', self)
        self.button9.clicked.connect(self.nautilus_models)
        self.button9.setToolTip('Start')

        self.button10 = QPushButton('PERF', self)
        self.button10.clicked.connect(self.nautilus_perf)
        self.button10.setToolTip('Start')

        self.button11 = QPushButton('DEL', self)
        self.button11.clicked.connect(self.delete_perf2_folder)
        self.button11.setToolTip('Start')
        self.button11.setStyleSheet("background-color: red")

        self.button12 = QPushButton('UPDATE', self)
        self.button12.clicked.connect(self.update_textfields)
        self.button12.setToolTip('Start')
        self.button12.setStyleSheet("background-color: yellow")

        self.button13 = QPushButton('CHANGE', self)
        self.button13.clicked.connect(self.update_time_steps)
        self.button13.setToolTip('Start')
        self.button13.setStyleSheet("background-color: yellow")

    def create_checkbox(self):
        #Checkbox for one or two step anctinf
        self.checkbox_one_step = QCheckBox("One-step", self)
        self.checkbox_one_step.stateChanged.connect(self.callback1)
        self.checkbox_two_step = QCheckBox("two-step", self)
        self.checkbox_two_step.stateChanged.connect(self.callback2)
        self.checkbox_one_step.setChecked(True)
        self.checkbox_two_step.setChecked(True)

        self.checkbox_one_kill = QCheckBox("Del Perf1", self)
        self.checkbox_one_kill.stateChanged.connect(self.callback3)
        self.checkbox_two_kill = QCheckBox("Del Perf2", self)
        self.checkbox_two_kill.stateChanged.connect(self.callback4)
        self.checkbox_one_kill.setChecked(True)
        self.checkbox_two_kill.setChecked(True)


    def move_rest(self):
        self.checkbox_one_step.move(self.x_rest_position + 270, 20)
        self.checkbox_two_step.move(self.x_rest_position + 270, 40)

        self.checkbox_one_kill.move(self.x_rest_position + 270, 80)
        self.checkbox_two_kill.move(self.x_rest_position + 270, 100)

        self.button.move(self.x_rest_position, 20)
        self.button2.move(self.x_rest_position + 110, 20)
        self.button3.move(self.x_rest_position, 60)
        self.button4.move(self.x_rest_position, 100)
        self.button5.move(self.x_rest_position, 140)
        self.button6.move(self.x_rest_position, 180)
        self.button7.move(self.x_rest_position + 320, 490)
        self.button8.move(self.x_rest_position + 320, 330)
        self.button9.move(self.x_rest_position + 320, 140)
        self.button10.move(self.x_rest_position + 320, 180)
        self.button11.move(self.x_rest_position + 320, 650)
        self.button12.move(self.x_rest_position + 290, 300)
        self.button13.move(self.x_rest_position + 290, 270)

    def resize_buttons(self):
        self.button.resize(100,30) #run
        self.button2.resize(140,30) #browse
        self.button3.resize(100,30) #load
        self.button4.resize(100,30)#plot
        self.button5.resize(100,30)#
        self.button6.resize(160, 30)
        self.button7.resize(40, 30)
        self.button8.resize(40, 30)
        self.button9.resize(40, 30)
        self.button10.resize(40, 30)
        self.button11.resize(40, 30)
        self.button12.resize(70, 20)
        self.button13.resize(70, 20)

    def open_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            os.system('cp ' + fileName + ' ' + './compare_models/')
            self.update_textfields()

        else:
            return

    #start actinf and retrieve data
    def create_data(self):
        a = 0
        if self.checkbox_one_step: a = 1
        if self.checkbox_two_step: b = 1
        full_percent = a * len(self.models) + b * len(self.models)
        counter = 0
        for model in self.models:
            id = self.models.index(model)
            if self.checkbox_one_step:
                os.system('python3 ./compare_actinf.py ' + model + ' ' + str(id) + ' ' + '1step' + ' ' + str(self.time_steps))
                counter += 1
                self.progress.setValue(int(counter/full_percent * 100))
                self.play_label(counter)

            if self.checkbox_two_step:
                os.system('python3 ./compare_actinf.py ' + model + ' ' + str(id) + ' ' + '2step'  + ' ' + str(self.time_steps))
                counter += 1
                self.progress.setValue(int(counter / full_percent * 100))
                self.play_label(counter)

    def reset_performance(self):
        if self.kill_d1:
            os.system('rm ./datas/* 2> /dev/null')
        if self.kill_d2:
            os.system('rm ./datas2/* 2> /dev/null')
        self.update_textfields()

    def plot_data(self):
        data1 = [[] for i in range(6)]
        data2 = [[] for i in range(6)]
        if self.plot_one_step:
            data1 = self.datas1
        if self.plot_two_step:
            data2 = self.datas2
            data2 = self.datas2

        self.m1.plot(data1[0], data2[0], 1)
        self.m2.plot(data1[1], data2[1],2)
        self.m3.plot(data1[2], data2[2],3)
        self.m4.plot(data1[3], data2[3],4)
        self.m5.plot(data1[4], data2[4],5)
        self.m6.plot(data1[5], data2[5],6)


    def load_data(self):
        if self.plot_one_step:
            for d in range(self.data1_items):
                d1 = np.load('./datas/' + str(d) + '-1' + '.npy')
                self.datas1[d] = d1
        if self.plot_two_step:
            for d in range(self.data2_items):
                d2 = np.load('./datas2/' + str(d) + '-2' + '.npy')
                self.datas2[d] = d2

    def load_datafrom_folder(self):
        print(self.models)
        model_string = ''
        os.system('ls ./datas/ > isghflksazdhgfgsdg.txt')
        os.system('ls ./datas2/ > aousdhasioudhoas.txt')
        path = Path('./')
        with open('isghflksazdhgfgsdg.txt', 'r') as infile:
            lines1 = [path + line.strip() for line in infile]
        infile.close()

        with open('aousdhasioudhoas.txt', 'r') as infile:
            lines2 = [path + line.strip() for line in infile]
        infile.close()

        for lin1 in lines1:
            data = np.load('./datas/' + lin1) #two step inf
            self.datas[id] = data

        for lin2 in lines2:
            data2 = np.load('./datas/' + str(id) + '-1' + '.npy') #one step inf
            self.datas2[id] = data2

    def update_text(self, id):
        if id == 1:
            txt = self.make_string(self.get_model_folder_inhalt())
            self.label1.setText(txt)
            self.show()
        if id == 2:
            txt = self.make_string(self.get_perf_folder_inhalt())
            if txt == '':
                self.checkbox_one_step.setChecked(False)
            self.label6.setText(txt)
            self.show()
        if id == 3:
            txt = self.make_string(self.get_perf2_folder_inhalt())
            if txt == '':
                self.checkbox_two_step.setChecked(False)
            self.label2.setText(txt)
            self.show()

    def get_model_folder_inhalt(self):
        os.system('ls ./compare_models/ > asvdhjgvsahjgd.txt')
        path = ('./compare_models/')
        with open('./asvdhjgvsahjgd.txt', 'r') as infile:
            lines = [path + line.strip() for line in infile]

        os.system('rm asvdhjgvsahjgd.txt')

        return lines

    def get_perf_folder_inhalt(self):
        os.system('ls ./datas/ > 4576245624356.txt')
        path = ('./compare_models/')
        with open('./4576245624356.txt', 'r') as infile:
            lines = [path + line.strip() for line in infile]
        infile.close()
        self.data1_items = len(lines)
        os.system('rm 4576245624356.txt')
        return lines

    def get_perf2_folder_inhalt(self):
        os.system('ls ./datas2/ > eisuyfbvlkuesfy.txt')
        path = ('./compare_models/')
        with open('./eisuyfbvlkuesfy.txt', 'r') as infile:
            lines = [path + line.strip() for line in infile]
        infile.close()
        self.data2_items = len(lines)
        os.system('rm eisuyfbvlkuesfy.txt')
        return lines

    @pyqtSlot()
    def update_time_steps(self):
        self.time_steps = int(self.label7.text())
        self.label8.setText('TIME STEPS: current = ' + str(self.time_steps))

    @pyqtSlot()
    def delete_perf_folder(self):
        os.system('rm ./datas/* 2> /dev/null') #2> /dev/null')
        self.datas = []
        self.update_textfields()

    @pyqtSlot()
    def delete_perf2_folder(self):
        os.system('rm ./datas2/* 2> /dev/null')  # 2> /dev/null')
        self.datas2 = []
        self.update_textfields()

    @pyqtSlot()
    def delete_mod_folder(self):
        os.system('rm ./compare_models/* 2> /dev/null')
        self.models = []
        self.update_textfields()

    @pyqtSlot()
    def nautilus_models(self):
        os.system('nautilus ./compare_models/ 2> /dev/null')

    @pyqtSlot()
    def nautilus_perf(self):
        os.system('nautilus ./datas/ 2> /dev/null')

    @pyqtSlot()
    def get_models_out_of_folder(self):
        print(self.plot_one_step)
        print(self.plot_two_step)
        lines = self.get_model_folder_inhalt()
        self.models = lines
        self.reset_performance()
        self.create_data()
        self.load_data()
        self.checkbox_one_step.setChecked(True)
        self.checkbox_two_step.setChecked(True)
        self.plot_from_folder()
        self.update_textfields()

    @pyqtSlot()
    def plot_from_folder(self):
        os.system('ls ./compare_models/ > asvdhjgvsahjgd.txt')
        path = ('./compare_models/')
        with open('./asvdhjgvsahjgd.txt', 'r') as infile:
            lines = [path + line.strip() for line in infile]
        infile.close()
        self.models = lines
        self.update_text(1)
        self.load_data()
        self.plot_data()

    @pyqtSlot()
    def on_click_run(self):
        print(self.plot_one_step)
        print(self.plot_two_step)
        lines = self.get_model_folder_inhalt()
        self.models = lines
        self.reset_performance()
        self.create_data()
        self.load_data()
        self.checkbox_one_step.setChecked(True)
        self.checkbox_two_step.setChecked(True)
        self.update_textfields()

    @pyqtSlot()
    def on_click_test(self):
        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 0)
        self.show()

    @pyqtSlot()
    def on_click_file_browse(self):
        self.open_data()

    @pyqtSlot()
    def on_click_file_load(self):
        self.load_data()

    @pyqtSlot()
    def on_click_plot(self):
        lines = self.get_model_folder_inhalt()
        self.models = lines
        self.load_data()
        self.plot_data()

    def callback1(self, state):
        if state == QtCore.Qt.Checked:
            self.plot_one_step = True
        else:
            self.plot_one_step = False

    def callback2(self, state):
        if state == QtCore.Qt.Checked:
            self.plot_two_step = True
        else:
            self.plot_two_step = False


    def callback3(self, state):
        if state == QtCore.Qt.Checked:
            self.kill_d1 = True
        else:
            self.kill_d1 = False

    def callback4(self, state):
        if state == QtCore.Qt.Checked:
            self.kill_d2 = True
        else:
            self.kill_d2 = False


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
        #self.plot()


    def plot(self, data, data2, id):
        if self.used:
            self.figure.clear()
        self.used = True

        ax = self.figure.add_subplot(111)
        ax.set_xlabel('time steps')
        ax.set_ylabel('distance from A to B')
        ax.grid(True)
        #ax.frame_on(True)
        lines = []
        names = []
        red_patch = mpatches.Patch(color='red', label='1 step')
        green_patch = mpatches.Patch(color='green', label='2 step')
        min_patch = mpatches.Patch(color='yellow', label='2 step')
        max_patch = mpatches.Patch(color='yellow', label='2 step')
        mean_patch = mpatches.Patch(color='yellow', label='2 step')
        patches = []
        if len(data) > 0:
            line1 = ax.plot(data, 'r-', label='one step')
            lines.append(line1)
            names.append('one step')
            ax.set_title(str(id))
            patches.append(red_patch)
        if len(data2) > 0:
            line2 = ax.plot(data2, 'g-', label='two step')
            lines.append(line2)
            names.append('two step')
            ax.set_title(str(id))
            patches.append(green_patch)
        else:
            ax.set_title('Not used')

        ax.legend(handles=patches)
        self.draw()



if __name__ == "__main__":
    s = Start_compare()
