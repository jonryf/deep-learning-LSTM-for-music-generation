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
    songs = songs.split(song_delimiter)
    songs = [song + song_delimiter for song in songs]

    return songs



def main():
    char_to_ix, ix_to_char = char_mapping()
    print(char_to_ix)


main()
