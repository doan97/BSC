import numpy as np
import matplotlib.pyplot as plt

import global_config as c

class SensorPlot(object):
    def __init__(self, size, title="Test", color="red"):

        bottom = 0
        max_height = 1

        width = (2*np.pi) / size
        offset = width / 2.
        theta = np.linspace(offset, 2 * np.pi + offset, size, endpoint=False)
        radii = np.empty((size,))
        

        self.fig, self.ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
        self.ax.set_ylim([0., 1.])
        self.sensor_data_bars = self.ax.bar(theta, radii, width=width, bottom=bottom, edgecolor='black', fill=False, linewidth=2)
        
        if c.OUTPUT_SENSOR_DIM > 0:
            self.sensor_prediction_bars = self.ax.bar(theta, radii, width=width, bottom=bottom, edgecolor=color, fill=False, linewidth=2)

        # plt.show()
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()


    def update(self, sensor_data, sensor_predictions, title=None):

        if title is not None:
            self.ax.set_title(title)
        else:
            self.ax.set_title("")
        
        [self.sensor_data_bars[i].set_height(sensor_data[i]) for i in range(len(self.sensor_data_bars))]

        if c.OUTPUT_SENSOR_DIM > 0:
            [self.sensor_prediction_bars[i].set_height(sensor_predictions[i]) for i in range(len(self.sensor_prediction_bars))]

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.show()

    def save(self, name):
        self.fig.savefig("./sensorplots/"+name+".pdf")

# s = SensorPlot(16)
# sensor_data = np.zeros([16])
# sensor_data[15] = 0.6
# s.update(sensor_data, np.zeros([16]))