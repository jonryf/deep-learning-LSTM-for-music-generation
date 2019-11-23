import math

from torch.utils.data.dataset import Dataset


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
