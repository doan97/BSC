import torch
import torch.nn as nn

# PyTorch network class
class Net(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, batch_size=1, num_layers=1):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)#,dropout=0.5)
        self.dense = nn.Linear(hidden_dim, output_dim)


    #i: input, o: output, h: hidden outs, c:
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),     # h (output to next cell)
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))     # c (inner Cell Value)

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of all_h: [input_size, batch_size, hidden_dim]
        # shape of last_output: (h, c), where h and c both
        # have shape (num_layers, batch_size, hidden_dim).
        all_h, last_output = self.lstm(input)

        last_h = last_output[0]
        last_c = last_output[1]
        
        all_predictions = self.dense(all_h[:, -1, :])

        return all_predictions, last_output
