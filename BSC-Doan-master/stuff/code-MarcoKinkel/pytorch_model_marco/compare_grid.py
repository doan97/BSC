import sys

import PyQt5

from PyQt5.QtCore import pyqtSlot
from compare_tools import Tools
from matplotlib.backends.qt_compat import QtWidgets


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.tools = Tools(self)
        self.tools.create_buttons_and_trees()
        self.tools.create_plot_canvas()
        self.tools.make_checkboxes_and_progress()
        #self.tools.dynamic_plot_canvas(10)
        self.give_buttons_onclick()


    @pyqtSlot()
    def on_click_run(self):
        self.tools.run_actinf()

    @pyqtSlot()
    def on_click_plot(self):
        #self.tools.plot_data()
        self.tools.dynamic_plot_data()
        self.tools.plot_data()

    @pyqtSlot()
    def on_click_train(self):
        self.tools.train('test_dropout')

    @pyqtSlot()
    def on_click_train_multiple_noise(self):
        self.tools.train_multiple_noise()

    @pyqtSlot()
    def on_click_train_multiple_dropout(self):
        self.tools.train_multiple_dropout()

    @pyqtSlot()
    def on_click_train_multiple_nd(self):
        self.tools.train_multiple_nd()

    def give_buttons_onclick(self):
        self.tools.b_test.clicked.connect(self.on_click_run)
        self.tools.b_plot.clicked.connect(self.on_click_plot)
        self.tools.b_train.clicked.connect(self.on_click_train)

        self.tools.b_mtrain_n.clicked.connect(self.on_click_train_multiple_noise)
        self.tools.b_mtrain_d.clicked.connect(self.on_click_train_multiple_dropout)
        self.tools.b_mtrain_nd.clicked.connect(self.on_click_train_multiple_nd)





if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()