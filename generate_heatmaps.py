import torch
from models import LSTMSimple
from utils import get_device, char_mapping, to_onehot
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

char_to_idx, idx_to_char = char_mapping()

config = {
    "VOCAB_SIZE": len(char_to_idx.keys()),
    "HIDDEN": 100,

    # For songs sampling
    "TEMPERATURE": 1,
    "TAKE_MAX_PROBABLE": False,
    "LIMIT_LEN": 440
}

MODEL_INPUT = "$\nX:3"
model = LSTMSimple(config["VOCAB_SIZE"], config["HIDDEN"], config["VOCAB_SIZE"]).to(get_device())
model.init_state()
model.load_state_dict(torch.load("trained_models/model2019-11-26-03-35.pth", map_location='cpu'))

model.eval()

text = """$
X:3
T:Trow Faicstieu
C:Itt
R:polka
Z:id:hn-hornpipe-59
M:C|
K:A
^GG|B2B B2c BGA|B2d c2c d2B|g6 A3|BdB dBA B2d|edf ecA Bdg|gdc AAF |1 dfdd g2ge ||
d2B2 BdcB|A2Bc BdAB|cBc BAFG|A~G2 d2e2|dBc BGF|cAF G2G:|
,2D2B c2GB|A2FA FABc|c2GF A2AB|1 cded dFGA|B2de egec|
A2GG B2cA|DBcd eABA|Bcde dBAB|A2Ac d2d2|cgfe cedc|=cBce fedc|AFEA GGAF|BEFG A2AB|(c2df (ggfa | 
a4d2c|a2g'ggf edBc|c2d2e2A2|
BcdBBc d2d2|d2f2 f2ca|f2faf2d2|c2c2d2d2|f2fd g4f2|2fedc 
"""
print(text[:440])

with torch.no_grad():

    values = []
    actual_letters = []

    for c in text:
        inputs_onehot = to_onehot(torch.Tensor([char_to_idx[c]]), len(char_to_idx.keys()))
        out = model(inputs_onehot.unsqueeze(1))
        out.squeeze_(1)
        value = model.state[0].view(-1)
        values.append([value[n].item() for n in range(100)])

        # If special character
        if c is '\n':
            c = 'nl'
        elif c is ' ':
            c = 'sp'
        actual_letters.append(c)

    for idx in range(100):
        data = np.reshape([value[idx] for value in values][:440], (20, -1))
        letters = np.reshape(actual_letters[:440], (20, -1))

        # Create heatmap
        heatmap = sns.heatmap(data, linewidth=0.5, cmap="RdBu_r")
        heatmap.invert_yaxis()
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                plt.text(x + 0.5, y + 0.5, letters[y][x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         )
        print("Saving image: " + str(idx))
        plt.savefig('heatmaps/hidden_neuron_' + str(idx) + '.png')
        plt.clf()
