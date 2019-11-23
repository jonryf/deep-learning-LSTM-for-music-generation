import math

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from models import LSTMSimple
from utils import SlidingWindowLoader, read_songs_from, char_mapping

# Check if cuda is supported
if torch.cuda.is_available():
    print("CUDA supported")
    computing_device = torch.device("cuda")
else:
    print("CUDA not supported")
    computing_device = torch.device("cpu")


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

# Load Data
char_to_idx, idx_to_char = char_mapping()

train, train_encoded = load_data('train.txt')
val, val_encoded = load_data('val.txt')

# Initialize model
VOCAB_SIZE = len(char_to_idx.keys())
EPOCHS = 10
CHUNK_SIZE = 100

model = LSTMSimple(VOCAB_SIZE, 100, VOCAB_SIZE)

criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
validation_losses = []

for epoch in range(EPOCHS):
    print("Epoch", epoch)
    training_error = []
    model.train()
    for i, song in enumerate(train_encoded):
        #print("Song: ", i)
        optimizer.zero_grad()
        p = 0
        n = math.ceil(len(song) / CHUNK_SIZE)
        loss = 0

        # Reset H for each song
        model.init_h()

        # Divide songs into chunks
        for mini in range(n):
            if p + CHUNK_SIZE + 1 > len(song):
                inputs = song[p:-1]
                targets = song[p + 1:]
            else:
                inputs = song[p:p + CHUNK_SIZE]
                targets = song[p + 1: p + CHUNK_SIZE + 1]
            p += CHUNK_SIZE

            # Skip if empty
            if inputs.size()[0] == 0:
                continue

            # One-hot chunk tensor
            inputs_onehot = to_onehot(inputs)

            # Forward through model
            output = model(inputs_onehot.unsqueeze(1))  # Turn input into 3D (chunk_length, batch, vocab_size)

            # Calculate
            output.squeeze_(1)  # Back to 2D

            loss += criterion(output, targets.long())

            ## Detatch H

        training_error.append(loss.item() / n)
        loss.backward()

        optimizer.step()
    print("Training Error: ", sum(training_error) / len(training_error))

    with torch.no_grad():
        model.eval()
        validation_song_losses = []
        for i, song in enumerate(val_encoded):
            print("Song: ", i)
            optimizer.zero_grad()
            p = 0
            n = math.ceil(len(song) / CHUNK_SIZE)
            loss = 0

            song_loss = 0

            for mini in range(n):
                if p + CHUNK_SIZE + 1 > len(song):
                    inputs = song[p:-1]
                    targets = song[p + 1:]
                else:
                    inputs = song[p:p + CHUNK_SIZE]
                    targets = song[p + 1: p + CHUNK_SIZE + 1]
                p += CHUNK_SIZE

                # Skip if empty
                if inputs.size()[0] == 0:
                    continue

                # One-hot chunk tensor
                inputs_onehot = to_onehot(inputs)

                # Forward through model
                output = model(inputs_onehot.unsqueeze(1))  # Turn input into 3D (chunk_length, batch, vocab_size)

                # Calculate
                output.squeeze_(1)  # Back to 2D

                song_loss += criterion(output, targets.long())

            validation_song_losses.append(song_loss.item())
        avg_val_loss = sum(validation_song_losses) / len(validation_song_losses)
        validation_losses.append(avg_val_loss)

        print('Epoch %d, Training loss: %.3d' % (epoch + 1, avg_val_loss))


