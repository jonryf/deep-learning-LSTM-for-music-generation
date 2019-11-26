import math

from torch.utils.data.dataset import Dataset
import torch


class SlidingWindowLoader(Dataset):
    def __init__(self, data, window=100):
        self.data = data
        self.window = window
        self.current = 0
        self.high = self.__len__()

    def __getitem__(self, index):
        index_pos = index * self.window
        x = self.data[index_pos:min(len(self.data) - 1, index_pos + self.window)]
        target = self.data[index_pos + 1:index_pos + 1 + self.window]
        return x, target

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if (self.current - 1) < self.high:
            return self.__getitem__(self.current - 1)
        raise StopIteration

    def __len__(self):
        return math.ceil(len(self.data) / self.window)


def check_cuda():
    if torch.cuda.is_available():
        print("CUDA supported")
        computing_device = torch.device("cuda")
    else:
        print("CUDA not supported")
        computing_device = torch.device("cpu")
    return computing_device


def encode_songs(songs, char_to_idx, computing_device):
    """
    Return a list of encoded songs where each char in a song is mapped to an index as in char_to_idx
    :param computing_device: cpu or cuda
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


def to_onehot(t, computing_device, vocab_size):
    """
    Take a list of indexes and return a one-hot encoded tensor
    :param vocab_size: Size of one hot encoding
    :param computing_device: cpu or cuda
    :param t: 1D Tensor of indexes
    :return: 2D Tensor
    """
    inputs_onehot = torch.zeros(t.shape[0], vocab_size).to(computing_device)
    inputs_onehot.scatter_(1, t.unsqueeze(1).long(), 1.0)  # Remember inputs is indexes, so must be integer
    return inputs_onehot


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
    chars.sort()  # To get the same order every time
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
    for x in range(10):
        char_to_ix, ix_to_char = char_mapping()
        print(char_to_ix)


#main()
