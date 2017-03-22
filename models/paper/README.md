## Parameters

### v1
steps_per_epoch on gru/lstm and gru2/lstm2 - 400
fade constant on gru/lstm and gru2/lstm2 - .2


### v2
steps_per_epoch on gru3/gru4 - 1000
fade constant on gru3/gru4 - .5


## Models

### GRU

```
model.add(GRU(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=(max_len, len(chars))))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(1))
model.add(PReLU())
```


### LSTM

```
model.add(LSTM(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=(max_len, len(chars))))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(1))
model.add(PReLU())
```


### GRU2

```
model.add(GRU(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=(max_len, len(chars))))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(64))
model.add(Dropout(.3))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(32))
model.add(Dropout(.3))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(1))
model.add(PReLU())
```


### LSTM2

```
model.add(LSTM(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=(max_len, len(chars))))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(64))
model.add(Dropout(.3))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(32))
model.add(Dropout(.3))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(1))
model.add(PReLU())
```


### GRU3

```
model.add( GRU(128, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=(max_len, len(chars))))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(64))
model.add(Dropout(.3))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(32))
model.add(Dropout(.3))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(1))
model.add(PReLU())
```


### GRU4

```
model.add( GRU(128, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=(max_len, len(chars))))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(128))
model.add(Dropout(.3))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(64))
model.add(Dropout(.3))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(32))
model.add(Dropout(.3))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(1))
model.add(PReLU())
```