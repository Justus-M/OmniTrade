import time

t = time.time()
import pandas as pd
import importlib
import TsDataProcessor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import tensorflow as tf
import random
import numpy as np

importlib.reload(TsDataProcessor)

predict = "BCH"
epochs = 10
Batch_size = 64
ValPercent = 0.05
hindsight = 60
foresight = 3
buy_threshhold= 0
y_name = "close"
x_names = ["close", "volume"]

cryptos = dict.fromkeys(["LTC", "BCH", "BTC", "ETH"])

for i in cryptos:
    cryptos[i] = pd.read_csv("Data/" + i + "-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'],index_col = "time", parse_dates=True)
    cryptos[i] = cryptos[i][x_names]
    cryptos[i].columns = i + " " + cryptos[i].columns.values

cryptos[predict]["target"] = cryptos[predict][predict + " " + y_name].shift(periods=-foresight) / cryptos[predict][predict + " " + y_name]
cryptos[predict]["target"] = (cryptos[predict]["target"] > 1 + buy_threshhold).astype(int)
cryptos[predict].dropna(inplace=True)

altcrypto = pd.DataFrame()

for i in cryptos:
    cryptos[i] = TsDataProcessor.Scaler(cryptos[i], "target")
    if i != predict:
        altcrypto[cryptos[i].columns.values] = cryptos[i]

combined, sequential = TsDataProcessor.TsDataProcessor(cryptos[predict], altcrypto, target = "target", t=hindsight)



days = sorted(combined.index.values)
valprop = days[-int(ValPercent*len(days))]

random.shuffle(sequential)
validation = combined[(combined.index >= valprop)]

validation = sequential[len(validation):]
train = sequential[:len(validation)]

validation = TsDataProcessor.Balance(validation)
train = TsDataProcessor.Balance(train)

train_x, train_y = TsDataProcessor.split(train)
val_x, val_y = TsDataProcessor.split(validation)

Model = Sequential()
Model.add(LSTM(128, input_shape =(train_x.shape[1:]), activation = "relu", return_sequences = True))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(LSTM(128, input_shape =(train_x.shape[1:]), activation = "relu", return_sequences = True))
Model.add(Dropout(0.1))
Model.add(BatchNormalization())

Model.add(LSTM(128, input_shape =(train_x.shape[1:]), activation = "relu"))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(Dense(32, activation="relu"))
Model.add(Dropout(0.2))

Model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

Model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

Model.fit(train_x, train_y, epochs = epochs, validation_data=(val_x, val_y))
