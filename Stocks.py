import time

t = time.time()
import pandas as pd
import importlib
import TsDataProcessor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import random
import numpy as np

importlib.reload(TsDataProcessor)

predict = "AAPL"
epochs = 10
Batch_size = 64
ValPercent = 0.05
hindsight = 60
foresight = 4096
buy_threshhold= 0
y_name = "close"
x_names = ["close", "volume"]

stocks = pd.read_csv("Data/all_stocks_5yr.csv", index_col = "date", parse_dates=True)
dex = pd.read_csv("Data/sentdex.csv", index_col = "date", parse_dates=True)

dex = dex[["sentiment_signal", "symbol"]]
stocks = stocks[["close", "volume", "Name"]]

stocks = stocks[stocks["Name"]!="APTV"]
dex = dex[dex["symbol"]==predict]

stocks = TsDataProcessor.StockFilter(stocks, tickercol = "Name")
dex = TsDataProcessor.StockFilter(dex, tickercol = "symbol")

stocks["target"] = stocks[predict + " " + y_name].shift(periods=-foresight) / stocks[predict + " " + y_name]
stocks["target"] = (stocks["target"] > 1 + buy_threshhold).astype(int)
stocks.dropna(inplace=True)

combined, sequential = TsDataProcessor.TsDataProcessor(stocks, dex, target = "target", t=hindsight)

days = sorted(combined.index.values)
valprop = days[-int(ValPercent*len(days))]

random.shuffle(sequential)
mark = combined[(combined.index >= valprop)]

validation = sequential[:len(mark)]
train = sequential[len(mark):]

train = TsDataProcessor.Balance(train)
validation = TsDataProcessor.Balance(validation)

train_x, train_y = TsDataProcessor.split(train)
val_x, val_y = TsDataProcessor.split(validation)

Model = Sequential()
Model.add(LSTM(128, input_shape =(train_x.shape[1:]), activation = "tanh", return_sequences = True))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(LSTM(128, input_shape =(train_x.shape[1:]), activation = "tanh", return_sequences = True))
Model.add(Dropout(0.1))
Model.add(BatchNormalization())

Model.add(LSTM(128, input_shape =(train_x.shape[1:]), activation = "tanh"))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(Dense(32, activation="relu"))
Model.add(Dropout(0.2))

Model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

Model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

Model.fit(train_x, train_y, batch_size = Batch_size,epochs = epochs, validation_data=(val_x, val_y))


