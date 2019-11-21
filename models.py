import torch
import torch.nn as nn

torch.manual_seed(42)


class LSTMSimple(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMSimple, self).__init__()
        self.hidden_size = hidden_size

        # Model layout
        self.embeddings = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, sequence):
        embedded = self.embeddings(sequence.view(1, -1))
        lstm_out, _ = self.lstm(embedded.view(1, 1, -1))
        return self.output(lstm_out.view(1, -1))
