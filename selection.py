from __future__ import print_function
from keras.models import Sequential, load_model
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

#
# How to fight with nans http://stackoverflow.com/questions/37232782/nan-when-training-regression-net-using-keras-and-theano
#

# After more than 100 iterations here are results for the first 20 receptors:
#6s - loss: 0.0566
#[[  5.0744682   11.84071827]
# [  5.22786273  11.8524065 ]
# [  5.50443683  11.84071827]
# [  6.25304755  11.89706039]
# [  6.33391     11.79355621]
# [  6.38307337  11.76680374]
# [  6.37860409  11.83020401]
# [  6.5170848   11.84064388]
# [  6.5273809   11.85927868]
# [  6.60817853  11.79740143]
# [  6.61521585  11.70992184]
# [  6.71463049  11.93083572]
# [  6.94914894  11.79218006]
# [  6.92190991  11.87408543]
# [  6.92383114  11.79092789]
# [  7.0442619   11.84865761]
# [  7.17130679  11.81984138]
# [  7.16884676  11.81451893]
# [  7.17130679  11.85754585]
# [  7.21923525  11.82061195]]
#
# Despite quite low loss, the proportions are looking bad. It seems that the optimizer
# "narrowed" the proportions in order to minimize loss (I checked - all proporions almost the same). 
# I'm pretty sure that such narrowing is due to the very heavy tail of receptors with 
# almost identical proportions (~ 11.8 in neg-log-scale).

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


if len(sys.argv) > 1:
    print("Loading model:", sys.argv[1])
    model = load_model(sys.argv[1])
else:
    model = Sequential()
    # Dropouts + BN works quite well. Without dropouts the learning process is faster
    # but I'm quite sure that this is due to the overtraining.
    model.add(LSTM(128, dropout_W = .2, dropout_U = .2, input_shape=(max_len, len(chars))))
    model.add(BatchNormalization())
    # Don't use L2 regularization ActivityRegularizer(l2 = .3) on the output Dense layer - 
    # it pushes the output to too strict boundaries, so the output will be always in [0,1]
    model.add(Dense(1))
    # Log-transorm the data and multiple by "-1" so it's possible to use ReLU instead of linear activations.
    model.add(Activation('relu'))

    # Very small learning rate because with the higher rate nans start to occur.
    optimizer = RMSprop(lr=0.000003)
    # Use MSLE instead of MSE because in case of log transformation of our data
    # the model will pay too much attention to optimization of differences for small
    # receptors, which has large negative (= large absolute value) values.
    model.compile(loss='msle', optimizer=optimizer)


print(model.summary())


for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, 
              batch_size=128, 
              nb_epoch=1, 
              verbose=VERBOSE,
              callbacks = [ModelCheckpoint(filepath = "model." + str(iteration % 2) + ".{epoch:02d}.hdf5")])

    a = y[:20].reshape((20,1))
    b = model.predict(X[:20,:,:])
    print(np.hstack([a, b]))
    
    a = y.reshape((30000,1))
    b = model.predict(X)
    print(np.hstack([a, b]))