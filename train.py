import math
import random

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from models import LSTMSimple
from utils import SlidingWindowLoader, read_songs_from, char_mapping
import matplotlib.pyplot as plt
import numpy as np


def encode_songs(songs, char_to_idx):
    """
    Return a list of encoded songs where each char in a song is mapped to an index as in char_to_idx
    :param songs: List[String]
    :param char_to_idx: Dict{char -> int}
    :return: List[Tensor]
    """
    songs_encoded = [0] * len(songs)
    for i, song in enumerate(songs):
        chars = list(song)
        result = torch.zeros(len(chars)).to(computing_device)
        for j, ch in enumerate(chars):
            result[j] = char_to_idx[ch]
        songs_encoded[i] = result
    return songs_encoded


def to_onehot(t):
    """
    Take a list of indexes and return a one-hot encoded tensor
    :param t: 1D Tensor of indexes
    :return: 2D Tensor
    """
    inputs_onehot = torch.zeros(t.shape[0], VOCAB_SIZE).to(computing_device)
    inputs_onehot.scatter_(1, t.unsqueeze(1).long(), 1.0)  # Remember inputs is indexes, so must be integer
    return inputs_onehot


def load_data(file):
    songs = read_songs_from('data/' + file)
    songs_encoded = encode_songs(songs, char_to_idx)
    return songs, songs_encoded


"""
Check for CUDA
"""
if torch.cuda.is_available():
    print("CUDA supported")
    computing_device = torch.device("cuda")
else:
    print("CUDA not supported")
    computing_device = torch.device("cpu")

"""
Load Data
"""
char_to_idx, idx_to_char = char_mapping()

train, train_encoded = load_data('train.txt')
val, val_encoded = load_data('val.txt')

"""
Initialize Model
"""
VOCAB_SIZE = len(char_to_idx.keys())
EPOCHS = 3
CHUNK_SIZE = 100

model = LSTMSimple(VOCAB_SIZE, 100, VOCAB_SIZE)
model.to(computing_device)

criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

"""
Train Model
"""
training_losses = []
validation_losses = []
for epoch in range(1, EPOCHS + 1):
    train_epoch_loss = []
    model.train()
    random.shuffle(train_encoded) # Shuffle songs for each epoch
    for i, song_encoded in enumerate(train_encoded):
        optimizer.zero_grad()

        # Reset H for each song
        model.init_h(computing_device)

        loss = 0
        n = 0
        for seq, target in SlidingWindowLoader(song_encoded):
            n += 1

            # if chunk is empty
            if len(seq) == 0:
                continue

            # One-hot chunk tensor
            inputs_onehot = to_onehot(seq)

            # Forward through model
            output = model(inputs_onehot.unsqueeze(1))  # Turn input into 3D (chunk_length, batch, vocab_size)

            # Calculate
            output.squeeze_(1)  # Back to 2D

            loss += criterion(output, target.long())

        avg_train_song_loss = loss.item() / n
        train_epoch_loss.append(avg_train_song_loss)  # Average loss over one song
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Song: {}, AvgTrainLoss: {}".format(i, sum(train_epoch_loss) / len(train_epoch_loss)))

    avg_train_songs_loss = sum(train_epoch_loss) / len(train_epoch_loss)  # Average loss overall songs
    training_losses.append(avg_train_songs_loss)

    with torch.no_grad():
        print("Validating...")
        model.eval()
        validation_song_losses = []
        loss = 0
        n = 0
        for seq, target in SlidingWindowLoader(song_encoded):
            n += 1

            # if chunk is empty
            if len(seq) == 0:
                continue

                # One-hot chunk tensor
                inputs_onehot = to_onehot(seq)

                # Forward through model
                output = model(inputs_onehot.unsqueeze(1))  # Turn input into 3D (chunk_length, batch, vocab_size)

                # Calculate
                output.squeeze_(1)  # Back to 2D

                song_loss += criterion(output, target.long())

            avg_val_song_loss = loss.item() / n
            validation_song_losses.append(avg_val_song_loss)
        avg_val_songs_loss = sum(validation_song_losses) / len(validation_song_losses)
        validation_losses.append(avg_val_songs_loss)

        print(
            "Epoch {}, Training loss: {}, Validation Loss: {}".format(epoch, avg_train_songs_loss, avg_val_songs_loss))

"""
Save Error plot
"""
x = np.arange(1, len(training_losses) + 1, 1)
plt.plot(x, training_losses, label="train loss")
plt.plot(x, validation_losses, label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.xticks(x)
plt.title("Loss as a function of number of epochs")
plt.legend()
plt.savefig('loss-plot.png')
