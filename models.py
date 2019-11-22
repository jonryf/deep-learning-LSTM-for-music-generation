import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(42)


class LSTMSimple(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMSimple, self).__init__()
        self.hidden_size = hidden_size

        # Model layout
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.h = None

    def init_h(self):
        self.h = (Variable(torch.zeros(1, 1, self.hidden_size)), Variable(torch.zeros(1, 1, self.hidden_size)))

    def forward(self, sequence):
        lstm_out, self.h = self.lstm(sequence, self.h)
        out = self.output(lstm_out)
        return out #F.log_softmax(out)

