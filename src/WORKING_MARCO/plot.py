import numpy as np
import matplotlib.pyplot as plt


plt.ion()
class Plot(object):
    def __init__(self, titles, ylims, xlims, title, linetype='-'):

        number_of_subplots = len(titles)
        self.xdata = []
        self.ydata = []
        self.lines = []

        for i in range(number_of_subplots):
            self.lines.append([])
            self.ydata.append([])

        self.fig, self.axes = plt.subplots(nrows=1, ncols=number_of_subplots, sharex=True, sharey=False, figsize=(10,5))

        self.fig.suptitle("Agent" + title)

        if number_of_subplots == 1:
            self.axes = [self.axes]

        for i in range(number_of_subplots):
            self.axes[i].set_title(titles[i])
            self.lines[i], = self.axes[i].plot([],[], linetype)

            if ylims is None:
                self.axes[i].set_autoscaley_on(True)
            else:
                self.axes[i].set_ylim(0., ylims[i])
        
            if xlims is None:
                self.axes[i].set_autoscalex_on(True)
            else:
                self.axes[i].set_xlim(0, xlims[i])

            # self.axes[i].axhline(y=0.01, color='lightgreen')
        
            #Other stuff
            self.axes[i].grid()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def plot(self, data, x_data=None, persist=False):
        # Plot
        
        if x_data is None:
            index = len(self.xdata)+1        
            self.xdata.append(index)
        else:
            self.xdata.append(x_data)

        for i in range(len(self.ydata)):
            # if(index == 9):
            #     self.axes[i].set_ylim(0., data[i])

            self.ydata[i].append(data[i])

            #Update data (with the new _and_ the old points)
            self.lines[i].set_xdata(self.xdata)
            self.lines[i].set_ydata(self.ydata[i])
        
            #Need both of these in order to rescale
            self.axes[i].relim()
            self.axes[i].autoscale_view()
        
        #We need to draw *and* flush
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # if persist:
        #     plt.show(block=True)

    def save(self, name):
        self.fig.savefig("./results/fig_"+name+".pdf")

    def show(self):
        plt.show(block=True)
