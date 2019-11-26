import math
import random
import time
import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from models import LSTMSimple, VanillaRNN
from utils import SlidingWindowLoader, read_songs_from, char_mapping, encode_songs, to_onehot, check_cuda
from generator import sample
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_data(file):
    songs = read_songs_from('data/' + file)
    songs_encoded = encode_songs(songs, char_to_idx, computing_device)
    return songs, songs_encoded


def negative_log_likelihood(model, encoded_data):
    """
    Average the cross entropy loss over all the chunks
    :param model: nn.Module
    :param encoded_data: List of encoded songs
    :return:
    """
    print("Calculating Negative Log Likelihood...")
    chunk_loss = 0
    number_of_chunks = 0
    with torch.no_grad():
        model.eval()
        for song_encoded in encoded_data:
            # Reset H for each song
            model.init_state(computing_device)

            for seq, target in SlidingWindowLoader(song_encoded):
                number_of_chunks += 1

                # if chunk is empty
                if len(seq) == 0:
                    continue

                # One-hot chunk tensor
                inputs_onehot = to_onehot(seq, computing_device, VOCAB_SIZE)

                # Forward through model
                output = model(inputs_onehot.unsqueeze(1))  # Turn input into 3D (chunk_length, batch, vocab_size)

                # Calculate
                output.squeeze_(1)  # Back to 2D

                chunk_loss += criterion(output, target.long())
    return chunk_loss / number_of_chunks


"""
Check for CUDA
"""
computing_device = check_cuda()

"""
Load Data
"""
char_to_idx, idx_to_char = char_mapping()

train, train_encoded = load_data('train.txt')
val, val_encoded = load_data('val.txt')
test, test_encoded = load_data('test.txt')

"""
Initialize Model
"""
VOCAB_SIZE = len(char_to_idx.keys())
EPOCHS = 10
CHUNK_SIZE = 100

# model = LSTMSimple(VOCAB_SIZE, 100, VOCAB_SIZE)
model = VanillaRNN(VOCAB_SIZE, 100, VOCAB_SIZE)
model.to(computing_device)

criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

"""
Train Model
"""
training_losses = []
validation_losses = []
for epoch in range(1, EPOCHS + 1):
    train_epoch_loss = []
    model.train()
    random.shuffle(train_encoded)  # Shuffle songs for each epoch
    for i, song_encoded in enumerate(train_encoded):
        # Reset H for each song
        model.init_state(computing_device)

        loss = 0
        n = 0  # Number of chunks made from a song
        for seq, target in SlidingWindowLoader(song_encoded):
            n += 1

            # if chunk is empty
            if len(seq) == 0:
                continue

            # One-hot chunk tensor
            inputs_onehot = to_onehot(seq, computing_device, VOCAB_SIZE)

            # Reset gradients for every forward
            optimizer.zero_grad()

            # Forward through model
            output = model(inputs_onehot.unsqueeze(1))  # Turn input into 3D (chunk_length, batch, vocab_size)

            output.squeeze_(1)  # Back to 2D

            chunk_loss = criterion(output, target.long())
            chunk_loss.backward()
            optimizer.step()
            loss += chunk_loss

        avg_train_song_loss = loss.item() / n
        train_epoch_loss.append(avg_train_song_loss)  # Average loss over one song

        if i % 100 == 0:
            print("Song: {}, AvgTrainLoss: {}".format(i, sum(train_epoch_loss) / len(train_epoch_loss)))

    avg_train_songs_loss = sum(train_epoch_loss) / len(train_epoch_loss)  # Average loss overall songs
    training_losses.append(avg_train_songs_loss)

    # Generate a song at this epoch
    song = sample(model, "$", 300, computing_device)
    print("-" * 40)
    print(song)
    print("-" * 40)

    with torch.no_grad():
        print("Validating...")
        model.eval()
        val_epoch_loss = []

        for i, song_encoded in enumerate(val_encoded):
            # Reset H for each song
            model.init_state(computing_device)

            loss = 0
            n = 0
            for seq, target in SlidingWindowLoader(song_encoded):
                n += 1

                # if chunk is empty
                if len(seq) == 0:
                    continue

                # One-hot chunk tensor
                inputs_onehot = to_onehot(seq, computing_device, VOCAB_SIZE)

                # Forward through model
                output = model(inputs_onehot.unsqueeze(1))  # Turn input into 3D (chunk_length, batch, vocab_size)

                # Calculate
                output.squeeze_(1)  # Back to 2D

                loss += criterion(output, target.long())

            avg_val_song_loss = loss.item() / n
            val_epoch_loss.append(avg_val_song_loss)
        avg_val_songs_loss = sum(val_epoch_loss) / len(val_epoch_loss)
        validation_losses.append(avg_val_songs_loss)

        print(
            "Epoch {}, Training loss: {}, Validation Loss: {}".format(epoch, avg_train_songs_loss, avg_val_songs_loss))

"""
Report NLL for validation and test
"""
nll_val = negative_log_likelihood(model, val_encoded)
nll_test = negative_log_likelihood(model, test_encoded)
print("NLL Validation: {}".format(nll_val))
print("NLL Test: {}".format(nll_test))

"""
Save Error plot
"""
x = np.arange(1, len(training_losses) + 1, 1)
plt.plot(x, training_losses, label="train loss")
plt.plot(x, validation_losses, label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.xticks(x)
plt.title("Loss as a function of number of epochs")
plt.legend()
plt.savefig('loss-plot.png')

"""
Save Model
"""
print("Saving model...")
now = datetime.now().strftime('%Y-%m-%d-%H-%M')
torch.save(model.state_dict(), "model" + now + ".pth")
print("Saved!")
