import torch
from models import VanillaRNN, LSTMSimple
from utils import get_device, char_mapping
from generator import sample

char_to_idx, idx_to_char = char_mapping()

config = {
    "VOCAB_SIZE": len(char_to_idx.keys()),
    "HIDDEN": 100,

    # For songs sampling
    "TEMPERATURE": 1,
    "TAKE_MAX_PROBABLE": False,
    "LIMIT_LEN": 300
}

MODEL_INPUT = "$\nX:3"
# MODEL_INPUT = "$"
model = LSTMSimple(config["VOCAB_SIZE"], config["HIDDEN"], config["VOCAB_SIZE"]).to(get_device())
model.init_state()
model.load_state_dict(torch.load("trained_models/model2019-11-26-00-41.pth", map_location='cpu'))
text = sample(model, MODEL_INPUT, config)
print(text)
