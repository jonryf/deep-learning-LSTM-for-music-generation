import math
from random import shuffle

import torch
from torch import optim
from torch.autograd import Variable
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


# Load Data
char_to_idx, idx_to_char = char_mapping()
songs = read_songs_from('data/train.txt')
songs_encoded = encode_songs(songs, char_to_idx)

# Initialize model
VOCAB_SIZE = len(char_to_idx.keys())
EPOCHS = 1
CHUNK_SIZE = 100

model = LSTMSimple(VOCAB_SIZE, 100, VOCAB_SIZE)

criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(EPOCHS):
    for song in songs_encoded:
        optimizer.zero_grad()
        p = 0
        n = math.ceil(len(song) / CHUNK_SIZE)
        loss = 0
        # Divide songs into chunks
        for mini in range(n):
            if p + CHUNK_SIZE > len(song):
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
        print(loss)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pass  # TODO
