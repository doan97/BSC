

from Net import Net
import torch
import torch.nn as nn
import torch.optim as optim
import global_config as c
import numpy as np
import datetime
class Model_Handler():
    def __init__(self, modelfile=None, datafile=None, cuda=False, act_inf=False):
        self.modelfile = modelfile
        self.use_cuda = cuda
        self.learning_rate = 0.01
        if act_inf:
            self.learning_rate = 0
            self.actinf_position_predictions = []

        self.load_model()
        if datafile is not None:
            self.load_data(datafile)

    def load_model(self):
        if self.use_cuda:
            self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(self.device)
            self.net = Net(24, 36, 20).to(self.device) #input, hidden, output
        else:
            self.net = Net(24, 36, 20) #input, hidden, output

        if self.modelfile is not None:
            self.net.load_state_dict(torch.load(self.modelfile))
            print('modelfile loaded.')

    def set_optimizer(self):
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, mode='min', factor=0.5, patience=10,
                                                              verbose=True, threshold=0.0001, threshold_mode='rel',
                                                              cooldown=0, min_lr=0, eps=1e-08)

    def predict(self, input):
        # Set initial state
        #self.net.hidden = self.actinf_previous_state

        # FWPass: predict path for current motor activity for one time step
        if self.use_cuda:
            input = input.to(self.device)
        prediction, state = self.net.forward(input)

        prediction = prediction.view(-1)
        position_prediction = prediction[c.OUTPUT_POSITION_DIM_START:c.OUTPUT_POSITION_DIM_END]
        self.actinf_position_predictions.append(position_prediction)

        self.actinf_previous_state = state

    def load_data(self, path):
        self.data = torch.load(path)
        print('data loaded')

    def create_random_input(self, i):
        input = self.data[i]
        return input

    def try_predict(self):
        start = datetime.datetime.now()
        for i in range(len(self.data)):
            data = self.create_random_input(i)
            self.predict(data)

        end = datetime.datetime.now()
        print(start, end)
        print(end-start)

handler = Model_Handler(modelfile='./compare_models/mode_T15_final.pt',datafile='./all_commands.pt', cuda=True, act_inf=True)
handler.try_predict()



