from random import shuffle

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from models import LSTMSimple
from utils import SlidingWindowLoader, read_songs_from, char_mapping

EPOCHS = 2
CHUNK_SIZE = 100

model = LSTMSimple(100, 100, 100)
songs = read_songs_from('data/train.txt')

char_to_idx, idx_to_char = char_mapping()
songs_encoded = []
for song in songs:
    result = [char_to_idx[ch] for ch in list(song)]
    songs_encoded.append(result)


criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

data_loaders = [SlidingWindowLoader(song, CHUNK_SIZE) for song in songs_encoded]

print(data_loaders)

for epoch in range(EPOCHS):
    shuffle(data_loaders)
    print("Epoch %d" % epoch )
    for song_loader in data_loaders:
        model.init_h()
        optimizer.zero_grad()
        loss = 0
        for index, song_chunk in enumerate(song_loader, 0):
            target = song_loader.get_target()
            output = model(song_chunk.float())
            print(target)

            loss += criterion(output, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pass  # TODO
