import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from models import LSTMSimple, VanillaRNN, fit, negative_log_likelihood, save_loss_graph
from utils import read_songs_from, char_mapping, encode_songs, get_device
from datetime import datetime


def load_data(file):
    songs = read_songs_from('data/' + file)
    songs_encoded = encode_songs(songs, char_to_idx)
    return songs, songs_encoded


char_to_idx, idx_to_char = char_mapping()

train, train_encoded = load_data('train.txt')
val, val_encoded = load_data('val.txt')
test, test_encoded = load_data('test.txt')

config = {
    "EPOCHS": 2,
    "CHUNK_SIZE": 100,
    "VOCAB_SIZE": len(char_to_idx.keys()),
    "LR": 0.001,  # Default in Adam 0.001
    "HIDDEN": 100,

    # For songs sampling
    "TEMPERATURE": 1,
    "TAKE_MAX_PROBABLE": False,
    "LIMIT_LEN": 300
}

# model = VanillaRNN(config["VOCAB_SIZE"], config["HIDDEN"], config["VOCAB_SIZE"]).to(get_device())
model = LSTMSimple(config["VOCAB_SIZE"], config["HIDDEN"], config["VOCAB_SIZE"]).to(get_device())

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters())

# Fit Model
fit(model, train_encoded, val_encoded, config)

# Report NLL for validation and test
nll_val = negative_log_likelihood(model, val_encoded, criterion, config)
nll_test = negative_log_likelihood(model, test_encoded, criterion, config)
print("NLL Validation: {}".format(nll_val))
print("NLL Test: {}".format(nll_test))

# Save error plot to file
save_loss_graph(model)

# Save model to file
print("Saving model...")
now = datetime.now().strftime('%Y-%m-%d-%H-%M')
torch.save(model.state_dict(), "model" + now + ".pth")
print("Saved!")
