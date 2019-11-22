import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class LSTMSimple(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMSimple, self).__init__()
        self.hidden_size = hidden_size

        # Model layout
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, sequence, h):
        lstm_out, h = self.lstm(sequence.view(1, 1, -1), h)
        out = self.output(lstm_out.view(1, -1))
        return F.softmax(out), h

