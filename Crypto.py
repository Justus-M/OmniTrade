import time

t = time.time()
import os
os.chdir("/Users/justusmulli/Projects/OmniTrade")
import pandas as pd
#import importlib
import TsDataProcessor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import random
import numpy as np

#importlib.reload(DataProcessor)
os.chdir("/Users/justusmulli/Data")
LTC = pd.read_csv("LTC-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'],index_col = "time", parse_dates=True)
BCH = pd.read_csv("BCH-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'],index_col = "time", parse_dates=True)
BTC = pd.read_csv("LTC-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'],index_col = "time", parse_dates=True)
ETH = pd.read_csv("ETH-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'],index_col = "time", parse_dates=True)

LTC.index = pd.to_datetime(LTC.index)
BCH.index = pd.to_datetime(BCH.index)
BTC.index = pd.to_datetime(BTC.index)
ETH.index = pd.to_datetime(ETH.index)

LTC["ticker"] = "LTC"
BCH["ticker"] = "BCH"
BTC["ticker"] = "BTC"
ETH["ticker"] = "ETH"


predict = "BTC"

epochs = 10
Batch_size = 64
ValPercent = 0.05

Main = TsDataProcessor.StockFilter(BTC, "close", "volume", tickercol ="ticker", target = predict, time = 3)
aBCH = TsDataProcessor.StockFilter(BCH, "close", "volume", tickercol ="ticker")
aETH = TsDataProcessor.StockFilter(ETH, "close", "volume", tickercol ="ticker")
aLTC = TsDataProcessor.StockFilter(LTC, "close", "volume", tickercol ="ticker")

combined, sequential = TsDataProcessor.TsDataProcessor(Main, aBCH, aETH, aLTC, target = predict, t=60)

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

lower = min(len(val_x)-sum(val_y), sum(val_y))



print(combined.head())

print(time.time() - t)

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
