import math
import sys

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from models import LSTMSimple
from utils import SlidingWindowLoader, read_songs_from, char_mapping

EPOCHS = 2
CHUNK_SIZE = 100

char_to_idx, idx_to_char = char_mapping()

model = LSTMSimple(len(char_to_idx) + 1, 100, len(char_to_idx) + 1)


def get_songs(data_set):
    songs = read_songs_from('data/' + data_set)

    songs_encoded = []
    for song in songs:
        result = []
        for ch in list(song):
            temp = [0] * (len(char_to_idx.keys()) + 1)
            temp[char_to_idx[ch]] = 1
            result.append(temp)
        songs_encoded.append(result)
    return songs, songs_encoded


criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#data_loaders = [SlidingWindowLoader(song, CHUNK_SIZE) for song in songs]

#print(data_loaders)

train, train_encoded = get_songs('train.txt')
val, val_encoded = get_songs('val.txt')

for epoch in range(EPOCHS):
    model.zero_grad()
    for song in train_encoded:
        p = 0
        n = math.ceil(len(song) / CHUNK_SIZE)
        loss = 0
        total = 0

        for mini in range(n):
            # Get input and target
            if p + CHUNK_SIZE > len(song):
                inputs = song[p:-1]
                targets = song[p + 1:]
            else:
                inputs = song[p:p + CHUNK_SIZE]
                targets = song[p + 1: p + CHUNK_SIZE + 1]
            p += CHUNK_SIZE

            inputs, targets = torch.Tensor(inputs), torch.Tensor(targets)
            total += len(inputs)

            #(sequence_len, batch, input_size)
            song_chunk = inputs.view(inputs.size()[0], 1, len(char_to_idx.keys()) + 1)

            targets = targets.view(inputs.size()[0], 1, len(char_to_idx.keys()) + 1)
            output = model(song_chunk)
            print(targets.max(dim=0)[1][0])
            output.squeeze_(1)

            loss += criterion(output, targets.max(dim=0))

            print('loss:', loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()



    with torch.no_grad():
        for song in val_encoded:
            pass


