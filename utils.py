import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


class SlidingWindowLoader(Dataset):
    def __init__(self, data, window):
        self.data = data  # torch.tensor(data)
        self.window = window
        self.current_index = 0

    def __getitem__(self, index):
        self.current_index = index
        x = self.data[index:index + self.window]
        if len(x) < 100:
            x = F.pad(input=x, pad=(0, 100 - len(x)), mode='constant', value=0)
        return x

    def __len__(self):
        return max(0, len(self.data) - self.window - 1)

    def get_target(self):
        slided_index = self.current_index + 1
        return self.data[slided_index:slided_index + self.window]


def char_mapping():
    """
    Mapping each unique char to an index for one-hot encoding
    :return: Dict{char -> index}, Dict{index -> char}
    """
    file = open("data/train.txt")
    text = file.read()
    text = text.replace("<start>", "$")
    text = text.replace("<end>", "%")
    chars = list(set(text))
    file.close()

    vocab_size = len(chars)
    print("Data has {} unique characters".format(vocab_size))

    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    return char_to_ix, ix_to_char


def read_songs_from(file_name):
    with open(file_name, 'r') as songs_file:
        songs = songs_file.read()
        songs = songs.replace("<start>", "$")
        songs = songs.replace("<end>", "%")
    song_delimiter = '%'
    songs = songs.split(song_delimiter)[:-1]
    songs = [song + song_delimiter for song in songs]
    return songs


def main():
    char_to_ix, ix_to_char = char_mapping()
    print(char_to_ix)

# main()
