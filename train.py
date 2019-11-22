import math
from random import shuffle

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from models import LSTMSimple
from utils import SlidingWindowLoader, read_songs_from, char_mapping

EPOCHS = 2
CHUNK_SIZE = 100

char_to_idx, idx_to_char = char_mapping()

model = LSTMSimple(len(char_to_idx) + 1, 100, 1)
songs = read_songs_from('data/train.txt')


songs_encoded = []
for song in songs:
    result = []
    for ch in list(song):
        temp = [0] * (len(char_to_idx.keys()) + 1)
        temp[char_to_idx[ch]] = 1
        result.append(temp)
    songs_encoded.append(result)


criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

data_loaders = [SlidingWindowLoader(song, CHUNK_SIZE) for song in songs]

print(data_loaders)

# for epoch in range(EPOCHS):
#     shuffle(data_loaders)
#     print("Epoch %d" % epoch )
#     song_n = 0
#     for song_loader in data_loaders:
#         model.init_h()
#         model.zero_grad()
#         loss = 0
#         for index, song_chunk in enumerate(song_loader, 0):
            #            target = song_loader.get_target()

for epoch in range(EPOCHS):
    for song in songs_encoded:
        p = 0
        n = math.ceil(len(song) / CHUNK_SIZE)
        loss = 0
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


            #(sequence_len, batch, input_size)
            song_chunk = inputs.view(100, 1, len(char_to_idx.keys()) + 1)
            print(song_chunk.size())
            targets = targets.view(100, 1, len(char_to_idx.keys()) + 1)
            output = model(song_chunk)

            _, index_targets = torch.max(targets, 2)
            loss += criterion(output, targets.view(100, 94))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pass  # TODO
