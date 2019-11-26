# PyTorch
import torch
from torch.distributions import Categorical
from torch.nn.functional import softmax

# Custom
from utils import char_mapping, encode_songs, to_onehot


def predict(model, song, config, char_to_idx, idx_to_char):
    """
    This function takes in the model and character as arguments and returns the next character prediction and hidden state.
    :param idx_to_char:
    :param char_to_idx:
    :param config: Dict of settings
    :param model: nn.Module
    :param song: String
    :return:
    """
    VOCAB_SIZE = len(char_to_idx.keys())

    encoded_song = encode_songs([song], char_to_idx)[0]
    inputs_onehot = to_onehot(encoded_song, VOCAB_SIZE)

    out = model(inputs_onehot.unsqueeze(1))
    out.squeeze_(1)

    prob = softmax(out[-1] / config["TEMPERATURE"], dim=0).data

    if config["TAKE_MAX_PROBABLE"]:
        char_ind = torch.max(prob, dim=0)[1].item()
    else:
        m = Categorical(prob)
        char_ind = m.sample().item()

    return idx_to_char[char_ind]


def sample(model, song, config):
    """
    # This function takes the desired output length and input characters as arguments, returning the produced sentence
    :param config: Dict of settings
    :param model: nn.Module
    :param song: String
    :param limit: Int
    :return: String (new generated song)
    """
    char_to_idx, idx_to_char = char_mapping()
    model.eval()

    i = 0
    while song[-1] != '%' and i < config["LIMIT_LEN"]:
        char = predict(model, song, config, char_to_idx, idx_to_char)
        song += char
        i += 1

    return song
