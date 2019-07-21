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

predict = "BTC"
epochs = 10
Batch_size = 64
ValPercent = 0.05
hindsight = 60
foresight = 3

cryptos = dict.fromkeys(["LTC", "BCH", "BTC", "ETH"])

for i in cryptos:
    cryptos[i] = pd.read_csv("Data/" + i + "-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'],index_col = "time", parse_dates=True)
    cryptos[i]["ticker"] = i

    if i == predict:
        cryptos[i] = TsDataProcessor.StockFilter(cryptos[i], "close", "volume", tickercol="ticker", target=predict, time=foresight)
    else:
        cryptos[i] = TsDataProcessor.StockFilter(cryptos[i], "close", "volume", tickercol="ticker")

target = cryptos[predict]
del cryptos[predict]

combined, sequential = TsDataProcessor.TsDataProcessor(target, cryptos["BCH"], cryptos["ETH"], cryptos["LTC"], target = predict, t=hindsight)

days = sorted(combined.index.values)
valprop = days[-int(ValPercent*len(days))]

random.shuffle(sequential)
validation = combined[(combined.index >= valprop)]

validation = sequential[0:len(validation)]
train = sequential[len(validation):]

buys = []
sells = []

for seq, target in validation:
    if target == 1:
        buys.append([seq, target])
    else:
        sells.append([seq, target])

lower = min(len(buys), len(sells))

validation = buys[:lower]+sells[:lower]

random.shuffle(validation)

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
