from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import ActivityRegularizer
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys
import re
import pandas as pd
import theano

VERBOSE=2
MAX_DATA = 30000

#theano.config.floatX="float32"
#theano.config.device="gpu1"
#theano.config.lib.cnmem="1."

R_exp_data = pd.read_table("azh.txt")
R_exp_data = R_exp_data[map(lambda x: not ("*" in x or "~" in x), R_exp_data["CDR3.amino.acid.sequence"])]
R_exp = R_exp_data["CDR3.amino.acid.sequence"][:MAX_DATA]
R_exp_prop = R_exp_data["Umi.proportion"][:MAX_DATA]
print("Experimental:\n -- #sequences:\t", len(R_exp), "\n -- #chars:\t", sum([len(x) for x in R_exp]))
print(" -- #nans:\t", R_exp.isnull().sum().sum())

#R_gen = list(pd.read_table("azh.aa.txt")["Amino.acid.sequence"])
#print("Generated:\n -- #sequences:\t", len(R_gen), "\n -- #chars:\t", sum([len(x) for x in R_gen]))


chars = ["A", "L", "R", 'K', 'N', 'M', 'D', 'F', 'C', 'P', 'Q', 'S', 'E', 'T', 'G', 'W', 'H', 'Y', 'I', 'V']
print('total chars:', len(chars))
print(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


#max_len = max(max([len(x) for x in R_exp]), max([len(x) for x in R_gen]))
max_len = max([len(x) for x in R_exp])

X = np.zeros((len(R_exp), max_len, len(chars)), dtype=np.bool)
y = - np.log(np.array(R_exp_prop))
for i, seq in enumerate(R_exp):
    for row, char in enumerate(seq):
        X[i, row, char_indices[char]] = 1


model = Sequential()
# Dropouts + BN works quite well. Without dropouts the learning process is faster
# but I'm quite sure that this due to the overtraining.
model.add(LSTM(128, dropout_W = .2, dropout_U = .2, input_shape=(max_len, len(chars))))
model.add(BatchNormalization())
# Don't use L2 regularization ActivityRegularizer(l2 = .3) on the output Dense layer - 
# it pushes the output to too strict boundaries, so the output will be always in [0,1]
model.add(Dense(1))
model.add(Activation('relu'))

# Very small learning rate because with the higher rate nans start to occur.
optimizer = RMSprop(lr=0.000003)
# Use MSLE instead of MSE because in case of log transformation of our data
# the model will pay too much attention to optimization of differences for small
# receptors, which has large negative (= large absolute value) values.
model.compile(loss='msle', optimizer=optimizer)

print(model.summary())

for iteration in range(1, 30):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, 
              batch_size=128, 
              nb_epoch=6, 
              verbose=VERBOSE,
              callbacks = [ModelCheckpoint(filepath = "model." + str(iteration % 2) + ".{epoch:02d}.hdf5")])

    a = y[:20].reshape((20,1))
    b = model.predict(X[:20,:,:])
    print(np.hstack([a, b]))