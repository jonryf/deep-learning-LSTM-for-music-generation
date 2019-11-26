# PyTorch
import torch
import torch.nn as nn
from torch.optim import Adam

# Custom
from utils import SlidingWindowLoader, to_onehot, get_device
from generator import sample

# Other
import matplotlib.pyplot as plt
import numpy as np
import random


class LSTMSimple(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMSimple, self).__init__()
        self.inputs_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Model layout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.state = None

        self.training_losses = []
        self.validation_losses = []

    def init_state(self):
        computing_device = get_device()
        cell = torch.zeros(1, 1, self.hidden_size).to(computing_device)
        hidden = torch.zeros(1, 1, self.hidden_size).to(computing_device)
        self.state = (cell, hidden)

    def forward(self, sequence):
        self.state[0].detach_()
        self.state[1].detach_()
        lstm_out, self.state = self.lstm(sequence, self.state)
        return self.fc(lstm_out)


class VanillaRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.inputs_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Model layout
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.state = None

        self.training_losses = []
        self.validation_losses = []

    def init_state(self):
        self.state = torch.zeros(1, 1, self.hidden_size).to(get_device())

    def forward(self, sequence):
        self.state.detach_()
        out, self.state = self.rnn(sequence, self.state)
        return self.fc(out)


def fit(model, train_encoded, val_encoded, config):
    """
    Fit the models weights and save the training and validation loss in the model
    :param model: nn. Module
    :param train_encoded: Encoded training data
    :param val_encoded: Encoded validation data
    :param config: dict with settings
    :return:
    """
    n_songs_train = len(train_encoded)
    n_songs_val = len(val_encoded)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["LR"])

    for epoch in range(1, config["EPOCHS"] + 1):
        train_loss = 0

        # Enter train mode to activate Dropout and Batch Normalization layers
        model.train()

        # Shuffle songs for each epoch
        random.shuffle(train_encoded)
        for i, song in enumerate(train_encoded):
            # Reset state for each song
            model.init_state()

            song_loss = 0
            n = 0  # Number of chunks made from song
            for seq, target in SlidingWindowLoader(song):

                # Chunks is sometimes empty
                if len(seq) == 0:
                    continue
                n += 1

                # One-hot encode chunk tensor
                input_onehot = to_onehot(seq, config["VOCAB_SIZE"])

                optimizer.zero_grad()  # Reset gradient for every forward
                output = model(input_onehot.unsqueeze(1))  # Size = (chunk_length, batch, vocab_size)
                output.squeeze_(1)  # Back to 2D
                chunk_loss = criterion(output, target.long())
                chunk_loss.backward()
                optimizer.step()
                song_loss += chunk_loss.item()
            train_loss += song_loss / n
            if i % 100 == 0:
                print("Song: {}, AvgTrainLoss: {}".format(i, train_loss / (i + 1)))

        # Append average training loss for this epoch
        model.training_losses.append(train_loss / n_songs_train)

        # Generate a song at this epoch
        song = sample(model, "$", config)
        print("{}\n{}\n{}".format("-" * 40, song, "-" * 40))

        # Validation
        with torch.no_grad():
            print("Validating")
            model.eval()  # Turns of Dropout and BatchNormalization
            val_loss = 0

            for song in val_encoded:
                # Reset state
                model.init_state()

                song_loss = 0
                n = 0
                for seq, target in SlidingWindowLoader(song):
                    # Chunks is sometimes empty
                    if len(seq) == 0:
                        continue
                    n += 1

                    # One-hot encode chunk tensor
                    input_onehot = to_onehot(seq, config["VOCAB_SIZE"])

                    output = model(input_onehot.unsqueeze(1))  # Size = (chunk_length, batch, vocab_size)
                    output.squeeze_(1)  # Back to 2D
                    song_loss += criterion(output, target.long()).item()
                val_loss += song_loss / n
            model.validation_losses.append(val_loss / n_songs_val)
            print("Epoch {}, Training loss: {}, Validation Loss: {}".format(epoch, model.training_losses[-1],
                                                                            model.validation_losses[-1]))


def negative_log_likelihood(model, encoded_data, criterion, config):
    """
    Average the cross entropy loss over all the chunks
    :param model: nn.Module
    :param encoded_data: List of encoded songs
    :return:
    """
    chunk_loss = 0
    number_of_chunks = 0
    with torch.no_grad():
        model.eval()
        for song_encoded in encoded_data:
            model.init_state()
            for seq, target in SlidingWindowLoader(song_encoded):
                number_of_chunks += 1
                if len(seq) == 0:
                    continue
                inputs_onehot = to_onehot(seq, config["VOCAB_SIZE"])
                output = model(inputs_onehot.unsqueeze(1))  # Turn input into 3D (chunk_length, batch, vocab_size)
                output.squeeze_(1)  # Back to 2D
                chunk_loss += criterion(output, target.long())
    return chunk_loss / number_of_chunks


def save_loss_graph(model):
    """
    Save the models training and validation loss plot to file
    :param model:
    :return: None
    """
    x = np.arange(1, len(model.training_losses) + 1, 1)
    plt.plot(x, model.training_losses, label="train loss")
    plt.plot(x, model.validation_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.xticks(x)
    plt.title("Loss as a function of number of epochs")
    plt.legend()
    plt.savefig('loss-plot.png')
