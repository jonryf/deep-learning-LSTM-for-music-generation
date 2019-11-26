import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMSimple(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMSimple, self).__init__()
        self.hidden_size = hidden_size

        # Model layout
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.h = None

    def init_h(self, computing_device):
        # (OUTPUT, HIDDEN)
        output = torch.zeros(1, 1, self.hidden_size).to(computing_device)
        hidden = torch.zeros(1, 1, self.hidden_size).to(computing_device)
        self.h = (output, hidden)

    def forward(self, sequence):
        self.h[0].detach_()
        self.h[1].detach_()
        lstm_out, self.h = self.lstm(sequence, self.h)
        out = self.output(lstm_out)
        return out
