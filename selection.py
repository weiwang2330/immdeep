from __future__ import print_function
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers import LSTM, GRU, Bidirectional, concatenate, average
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.metrics import binary_accuracy, binary_crossentropy
import numpy as np
from numpy.random import randint
import random
import sys
import re
import pandas as pd
import theano
from scipy import sparse
import os

#
# How to fight with nans http://stackoverflow.com/questions/37232782/nan-when-training-regression-net-using-keras-and-theano
#

VERBOSE=2
# MAX_DATA=10000
EPOCHS=300
BATCH_SIZE=64

#theano.config.floatX="float32"
#theano.config.device="gpu1"
#theano.config.lib.cnmem="1."


#################
# Load the data #
#################
R_exp_data = pd.read_table("data/base.txt")
R_exp_data = R_exp_data[np.array(list(map(lambda x: not ("*" in x or "~" in x), R_exp_data["CDR3.amino.acid.sequence"])))]
R_exp_data = R_exp_data.loc[R_exp_data["Umi.count"] > 1, :]
R_exp_seq = R_exp_data["CDR3.amino.acid.sequence"]
# R_exp_prop = R_exp_data["Umi.proportion"]
# R_exp_prop = R_exp_data["Umi.count"]
print("Experimental:\n -- #sequences:\t", len(R_exp_seq), "\n -- #chars:\t", sum([len(x) for x in R_exp_seq]))
print(" -- #nans:\t", R_exp_seq.isnull().sum().sum())


R_gen_data = pd.read_table("data/gen.txt", index_col = False)
R_gen_seq = R_gen_data["Amino.acid.sequence"]

print("Experimental:\n -- #sequences:\t", len(R_gen_seq), "\n -- #chars:\t", sum([len(x) for x in R_gen_seq]))
print(" -- #nans:\t", R_gen_seq.isnull().sum().sum())


######################
# Vectorize the data #
######################
def vectorize(seq_vec, class_sel, max_len, chars):
    X = np.zeros((len(seq_vec), max_len, len(chars)), dtype=np.bool)
    for i, seq in enumerate(seq_vec):
        for row, char in enumerate(seq):
            X[i, row, char_indices[char]] = 1
    return X, np.full((len(seq_vec), 1), class_sel)


chars = ["A", "L", "R", 'K', 'N', 'M', 'D', 'F', 'C', 'P', 'Q', 'S', 'E', 'T', 'G', 'W', 'H', 'Y', 'I', 'V']
print('total chars:', len(chars))
print(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


max_len = max(max([len(x) for x in R_exp_seq]), max([len(x) for x in R_gen_seq]))
print("max len:", max_len)

X_exp, y_exp = vectorize(R_exp_seq, 1, max_len, chars)
X_gen, y_gen = vectorize(R_gen_seq, 0, max_len, chars)


####################
# Load the CV data #
####################
val_names = ["top", "med", "low", "nei", "non", "put"]
val_Xs = []
val_ys = []

val_df = pd.read_table("data/val_1.txt")
X_val, y_val = vectorize(val_df["CDR3.amino.acid.sequence"], 1, max_len, chars)
val_Xs.append(X_val)
val_ys.append(y_val)

val_df = pd.read_table("data/val_2.txt")
X_val, y_val = vectorize(val_df["CDR3.amino.acid.sequence"], 1, max_len, chars)
val_Xs.append(X_val)
val_ys.append(y_val)

val_df = pd.read_table("data/val_3.txt")
X_val, y_val = vectorize(val_df["CDR3.amino.acid.sequence"], 1, max_len, chars)
val_Xs.append(X_val)
val_ys.append(y_val)

val_df = pd.read_table("data/val_4.txt", header = None)
X_val, y_val = vectorize(val_df[0], 0, max_len, chars)
val_Xs.append(X_val)
val_ys.append(y_val)

val_df = pd.read_table("data/val_5.txt", header = None)
X_val, y_val = vectorize(val_df[0], 0, max_len, chars)
val_Xs.append(X_val)
val_ys.append(y_val)

val_df = pd.read_table("data/val_67.txt", header = None)
X_val, y_val = vectorize(val_df[0], 0, max_len, chars)
val_Xs.append(X_val)
val_ys.append(y_val)


###################
# Build the model #
###################

dir_name = "models/" + sys.argv[1] + "/"

if not os.path.exists(dir_name):
    print("Creating '", dir_name, "'", sep="")
    os.makedirs(dir_name)
    
    
model = Sequential()

if len(sys.argv) > 2:
    if sys.argv[2].find("model") != -1:
        print("Loading model:", sys.argv[2])
        model = load_model(sys.argv[2])
    elif sys.argv[2] == "lstm":
        model.add(LSTM(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=(max_len, len(chars))))
        model.add(BatchNormalization())
        model.add(PReLU())
        
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(.3))
        
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(.3))
        
        model.add(Dense(1, activation = "sigmoid"))
        
    elif sys.argv[2] == "gru":
        model.add( GRU(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=(max_len, len(chars))))
        model.add(BatchNormalization())
        model.add(PReLU())
        
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(.3))
        
        # model.add(Dense(64))
        # model.add(BatchNormalization())
        # model.add(PReLU())
        # model.add(Dropout(.3))
        
        # model.add(Dense(32))
        # model.add(Dropout(.3))
        # model.add(PReLU())
        # model.add(BatchNormalization())
        
        model.add(Dense(1, activation = "sigmoid"))
        
    elif sys.argv[2] == "bigru":
        inp = Input(shape=(max_len, len(chars)))
        gru_node = lambda x: GRU(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                                 implementation=2, bias_initializer="he_normal",
                                 dropout=.2, recurrent_dropout=.2, unroll=True, go_backwards = x)
        forw = gru_node(False)(inp)
        forw = BatchNormalization()(forw)
        forw = PReLU()(forw)
        
        back = gru_node(True)(inp)
        back = BatchNormalization()(back)
        back = PReLU()(back)
        
        merged = average([forw, back])
        
        pred = Dense(64)(merged)
        pred = BatchNormalization()(pred)
        pred = PReLU()(pred)
        pred = Dropout(.3)(pred)
        
        pred = Dense(64)(pred)
        pred = BatchNormalization()(pred)
        pred = PReLU()(pred)
        pred = Dropout(.3)(pred)
        
        pred = Dense(1, activation = "sigmoid")(pred)
        
        model = Model(inp, pred)
        
    else:
        print("Unknown parameter:", sys.argv[2])
        sys.exit()


opt = Nadam()
model.compile(loss='binary_crossentropy', metrics = ["acc"], optimizer=opt)
print(model.summary())


####################
# Choose the batch #
####################
def generate_batch_simple(max_data, step):
    while True:
        indices_exp = randint(0, X_exp.shape[0], size=max_data // 2)
        indices_gen = randint(0, X_gen.shape[0], size=max_data // 2)
        yield np.vstack([X_exp[indices_exp], X_gen[indices_gen]]), \
              np.vstack([y_exp[indices_exp], y_gen[indices_gen]])
            
            
def generate_batch_weighted(max_data, step):
    while True:
        to_sample_big   = int(.8 * max_data)
        to_sample_small = max_data - to_sample_big
        indices_big   = randint(0, X_big.shape[0], size=to_sample_big)
        indices_small = randint(0, X_small.shape[0], size=to_sample_small)
        yield np.vstack([X_big[indices_big], X_small[indices_small]]), \
              np.vstack([y_big[indices_big], y_small[indices_small]]), \
              np.vstack([weights_big[indices_big], weights_small[indices_small]]).reshape((max_data,))
                
                
def generate_batch_top(max_data, step):
    while True:
        to_sample_big   = int(.8 * max_data) - 30
        to_sample_small = max_data - to_sample_big - 30
        indices_big   = np.concatenate([np.array(range(30)), randint(30, X_big.shape[0], size=to_sample_big)])
        indices_small = randint(0, X_small.shape[0], size=to_sample_small)
        yield np.vstack([X_big[indices_big], X_small[indices_small]]), \
              np.vstack([y_big[indices_big], y_small[indices_small]]), \
              np.vstack([weights_big[indices_big].astype(np.float), weights_small[indices_small].astype(np.float)]).reshape((max_data,))

                
def generate_batch_fading(max_data, step):
    while True:
        to_sample_big   = int(.8 * max_data)
        to_sample_small = max_data - to_sample_big
        indices_big   = randint(0, X_big.shape[0], size=to_sample_big)
        indices_small = randint(0, X_small.shape[0], size=to_sample_small)
        yield np.vstack([X_big[indices_big], X_small[indices_small]]), \
              np.vstack([y_big[indices_big], y_small[indices_small]]), \
              np.vstack([np.exp(np.log(weights_big[indices_big].astype(np.float)) / (step ** .5)), weights_small[indices_small].astype(np.float)]).reshape((max_data,))

                
     
generate_batch = generate_batch_simple
if len(sys.argv) > 3:
    if sys.argv[3] == "simple":
        generate_batch = generate_batch_simple
    elif sys.argv[3] == "wei":
        generate_batch = generate_batch_weighted
    elif sys.argv[3] == "fade":
        generate_batch = generate_batch_fading
    elif sys.argv[3] == "top":
        generate_batch = generate_batch_top


###################
# Train the model #
###################
epoch_per_iter = 5
for epoch in range(epoch_per_iter, EPOCHS+1, 5):
    history = model.fit_generator(generate_batch(BATCH_SIZE, epoch), 
                        steps_per_epoch=1000,
                        epochs=epoch, 
                        verbose=VERBOSE, 
                        initial_epoch=epoch - epoch_per_iter,
                        callbacks = [ModelCheckpoint(filepath = dir_name + "model." + str(epoch % 2) + ".hdf5"), 
                                     ReduceLROnPlateau(monitor="loss", factor=0.2, patience=3, cooldown=1, min_lr=0.0001)])

    for key in history.history.keys():
        with open(dir_name + "history." + key + ".txt", "a" if epoch > epoch_per_iter else "w") as hist_file:
            hist_file.writelines("\n".join(map(str, history.history[key])) + "\n")

    print("\nPredict R exp:\n  real\t\tpred")
    a = y_exp[:20].reshape((20,1))
    b = model.predict(X_exp[:20,:,:])
    print(np.hstack([a, b]), "\n")

    print("Predict R gen:\n  real\t\tpred")
    a = y_gen[-20:].reshape((20,1))
    b = model.predict(X_gen[-20:,:,:])
    print(np.hstack([a, b]))

    for i, vn in enumerate(val_names):
        y_true = val_ys[i]
        y_pred = model.predict(val_Xs[i])
        loss_val, acc_val = model.evaluate(val_Xs[i], val_ys[i], verbose=0)
        with open(dir_name + "history.loss.val_" + vn + ".txt", "a" if epoch > epoch_per_iter else "w") as hist_file:
            hist_file.writelines(str(loss_val) + "\n")
        with open(dir_name + "history.acc.val_" + vn + ".txt", "a" if epoch > epoch_per_iter else "w") as hist_file:
            hist_file.writelines(str(acc_val) + "\n")
            
            
for i, vn in enumerate(val_names):
    y_pred = model.predict(val_Xs[i])
    with open(dir_name + "pred.val_" + vn + ".txt", "wb") as pred_file:
        np.savetxt(pred_file, y_pred)
