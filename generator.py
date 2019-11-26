# PyTorch
import torch
from torch.distributions import Categorical
from torch.nn.functional import softmax

# Custom
from models import LSTMSimple
from utils import char_mapping, encode_songs, to_onehot, check_cuda

char_to_idx, idx_to_char = char_mapping()
VOCAB_SIZE = len(char_to_idx.keys())

TEMPERATURE = 1
TAKE_MAX_PROBABLE = False


def predict(model, song, computing_device):
    """
    This function takes in the model and character as arguments and returns the next character prediction and hidden state
    :param computing_device: cpu or cuda
    :param model: nn.Module
    :param song: String
    :return:
    """
    encoded_song = encode_songs([song], char_to_idx, computing_device)[0]
    inputs_onehot = to_onehot(encoded_song, computing_device, VOCAB_SIZE)

    out = model(inputs_onehot.unsqueeze(1))
    out.squeeze_(1)

    prob = softmax(out[-1] / TEMPERATURE, dim=0).data

    if TAKE_MAX_PROBABLE:
        char_ind = torch.max(prob, dim=0)[1].item()
    else:
        m = Categorical(prob)
        char_ind = m.sample().item()

    return idx_to_char[char_ind]


def sample(model, song, limit, computing_device):
    """
    # This function takes the desired output length and input characters as arguments, returning the produced sentence
    :param computing_device: cpu or cuda
    :param model: nn.Module
    :param song: String
    :param limit: Int
    :return: String (new generated song)
    """
    print("Start sample")
    model.eval()

    i = 0
    while song[-1] != '%' and i < limit:
        char = predict(model, song, computing_device)
        song += char
        i += 1

    return song


if __name__ == '__main__':
    MODEL_INPUT = "$\nX:3"
    # MODEL_INPUT = "$"
    computing_device = check_cuda()

    model = LSTMSimple(VOCAB_SIZE, 100, VOCAB_SIZE)
    model.to(computing_device)
    model.init_h(computing_device)
    model.load_state_dict(torch.load("trained_models/model2019-11-25-17-31.pth", map_location='cpu'))
    text = sample(model, MODEL_INPUT, 400, computing_device)
    print(text)
