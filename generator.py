import torch
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.nn.functional import softmax

from models import LSTMSimple
from utils import char_mapping

TEMPERATURE = 1
TAKE_MAX_PROBABLE = False


# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, char):
    char_to_idx, idx_to_char = char_mapping()

    encoded_char = char_to_idx[char]

    hidden = (Variable(torch.zeros(1, 1, 100)), Variable(torch.zeros(1, 1, 100)))
    out, hidden = model(encoded_char, hidden)

    prob = softmax(out[-1] / TEMPERATURE, dim=0).data

    if TAKE_MAX_PROBABLE:
        char_ind = torch.max(prob, dim=0)[1].item()
    else:
        m = Categorical(prob)
        char_ind = m.sample()

    return idx_to_char[char_ind], hidden


# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, start, limit):
    model.eval() # eval mode

    # First off, run through the starting characters
    chars = [ch for ch in start]

    # Now pass in the previous characters and get a new one
    i = 0
    last_five_chars = ''
    while last_five_chars != '<end>' and i < limit:
        char, h = predict(model, chars)
        chars.append(char)

        last_five_chars = ''.join(chars[-5:])
        i += 1

    return ''.join(chars)



def main():
    model_input = '<start>'

    model = LSTMSimple(94, 100, 94)
    sample(model, model_input, 20)


main()


