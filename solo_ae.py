from keras.layers import Input, LSTM, RepeatVector, Activation
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np
from numpy.random import randint
import pandas as pd
import os
import pprint

EPOCHS = 30
#train_size = 5000
test_size = 5

data = pd.read_table('all.txt')

maxlen = max(data["cdr3aa"].apply(len))

def align(seq):
    while len(seq) < maxlen:
        seq+='X'
    return seq

aligned = data.cdr3aa.apply(align)

#indices_tr = randint(len(aligned), size=train_size)
indices_t = randint(len(aligned), size=test_size)

#train = aligned[indices_tr]
test = aligned[indices_t]
train = aligned

chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def vectorization(seqs):
    X = np.zeros((len(seqs), maxlen, len(chars)), dtype=np.float)
    for i, sequence in enumerate(seqs):
        for t, char in enumerate(sequence):
            X[i, t, char_indices[char]] = 1
    return X

def devectorization(vecs):
    seqs = []
    for vec in vecs:
        seq = str()
        for pos in vec:
            seq += indices_char[np.argmax(pos)]
        seqs.append(seq)
    return seqs

x_train = vectorization(train)
x_test = vectorization(test)

dir_name = "autoencoders/ae64"

if not os.path.exists(dir_name):
    print("Creating '", dir_name, "'", sep="")
    os.makedirs(dir_name)

latent_dim = 64
timesteps = maxlen
input_dim = len(chars)

inputs = Input(shape=(timesteps, input_dim))
# encoded = Bidirectional(LSTM(latent_dim), merge_mode='sum')(inputs)
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)
activ = Activation('softmax')(decoded)

sequence_autoencoder = Model(inputs, activ)

encoder = Model(inputs, encoded)

sequence_autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(sequence_autoencoder.summary())

epoch_per_iter = 5
for epoch in range(epoch_per_iter, EPOCHS+1, 5):
    history = sequence_autoencoder.fit(x_train, x_train, epochs=epoch, batch_size=256, shuffle=True,
                                       initial_epoch=epoch-epoch_per_iter,
                                       callbacks=[ModelCheckpoint(filepath = dir_name + "model." + str(epoch % 5) + ".hdf5")])
    print(test)
    pprint.pprint(devectorization(sequence_autoencoder.predict(x_test)))
    print(sequence_autoencoder.predict(x_test)[0][0])
    for key in history.history.keys():
        with open(dir_name + "history." + key + ".txt", "a" if epoch > epoch_per_iter else "w") as hist_file:
            hist_file.writelines("\n".join(map(str, history.history[key])) + "\n")
