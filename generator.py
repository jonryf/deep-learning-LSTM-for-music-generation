import torch
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.nn.functional import softmax

from models import LSTMSimple
from utils import char_mapping

char_to_idx, idx_to_char = char_mapping()
VOCAB_SIZE = len(char_to_idx.keys())

TEMPERATURE = 1
TAKE_MAX_PROBABLE = True

if torch.cuda.is_available():
    print("CUDA supported")
    computing_device = torch.device("cuda")
else:
    print("CUDA not supported")
    computing_device = torch.device("cpu")


def encode_songs(songs, char_to_idx):
    """
    Return a list of encoded songs where each char in a song is mapped to an index as in char_to_idx
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


def to_onehot(t):
    """
    Take a list of indexes and return a one-hot encoded tensor
    :param t: 1D Tensor of indexes
    :return: 2D Tensor
    """
    inputs_onehot = torch.zeros(t.shape[0], VOCAB_SIZE).to(computing_device)
    inputs_onehot.scatter_(1, t.unsqueeze(1).long(), 1.0)  # Remember inputs is indexes, so must be integer
    return inputs_onehot


# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, song):

    encoded_song = encode_songs([song], char_to_idx)[0]
    inputs_onehot = to_onehot(encoded_song)

    out = model(inputs_onehot.unsqueeze(1))
    out.squeeze_(1)

    prob = softmax(out[-1] / TEMPERATURE, dim=0).data

    if TAKE_MAX_PROBABLE:
        char_ind = torch.max(prob, dim=0)[1].item()
    else:
        m = Categorical(prob)
        char_ind = m.sample().item()

    return idx_to_char[char_ind]


# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, song, limit):
    model.eval()
    model.init_h(computing_device)

    i = 0
    while song[-1] != '%' and i < limit:
        char = predict(model, song)
        song += char
        i += 1

    return song


MODEL_INPUT = """$
X:4
T:La gleiso de santo
O:France
C:?
A:Provence
Z:Transcrit et/ou corrig? par Michel BELLON - 2005-07-22
Z:Pour toute observation mailto:galouvielle@free.fr
M:C
Q:"Andante"
K:F
V:1
zGFc F2G2 | A6-Az | zA/B/ cA d2 cB | A6-Az | zGFc F2G2 | A6-Az | zA/B/ cA d2cB | A4-Az zc |:
cd cB AG"""

# MODEL_INPUT = "$"

def main():
    model = LSTMSimple(VOCAB_SIZE, 100, VOCAB_SIZE)
    model.to(computing_device)
    model.load_state_dict(torch.load("C:\\Users\\MichaelT\\Desktop\\model2019-11-23-16-20.pth", map_location='cpu'))
    text = sample(model, MODEL_INPUT, 300)
    print(text)

main()


