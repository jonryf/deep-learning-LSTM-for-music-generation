import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


class SlidingWindowLoader(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window
        self.current_index = 0

    def __getitem__(self, index):
        self.current_index = index
        x_data = self.data[index:index+self.window]
        x = torch.tensor(x_data)
        print(len(x))
        if len(x_data) < 100:
            x = F.pad(input=x, pad=(0, 100-len(x)), mode='constant', value=0)
        return x

    def __len__(self):
        return len(self.data) - self.window - 1

    def get_target(self):
        slided_index = self.current_index + 1
        return torch.tensor(self.data[slided_index:slided_index+1])



def char_mapping():
    """
    Mapping each unique char to an index for one-hot encoding
    :return: Dict{char -> index}, Dict{index -> char}
    """
    file = open("data/train.txt")
    chars = list(set(file.read()))  # TODO: Might have to tokenize <start> and <end>
    file.close()

    vocab_size = len(chars)
    print("Data has {} unique characters".format(vocab_size))

    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    return char_to_ix, ix_to_char


def read_songs_from(file_name):
    with open(file_name, 'r') as songs_file:
        songs = songs_file.read()

    song_delimiter = '<end>'
    songs = songs.split(song_delimiter)[:-1]
    songs = [song + song_delimiter for song in songs]

    return songs



def main():
    char_to_ix, ix_to_char = char_mapping()
    print(char_to_ix)


main()
