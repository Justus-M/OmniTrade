import time

t = time.time()
import os
os.chdir("/Users/justinmulli/PycharmProjects/Stocks")

import pandas as pd
import importlib
import DataProcessor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import random

importlib.reload(DataProcessor)

stocks = pd.read_csv("all_stocks_5yr.csv", index_col = "date", parse_dates=True)

dex = pd.read_csv("sentdex.csv", index_col = "date", parse_dates=True)

predict = "AAPL"

epochs = 100
Batch_size = 32
ValPercent = 0.2

Main = DataProcessor.StockFilter(stocks, "Name", "close", "volume", target = predict, time = 5)
Dex = DataProcessor.StockFilter(dex, "symbol", "sentiment_signal")

combined, sequential = DataProcessor.TsDataProcessor(Main, Dex, target = predict, t=30)

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

train_x, train_y = DataProcessor.split(train)
val_x, val_y = DataProcessor.split(validation)

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

print(sum("val hits "+ str(val_y)))
print(sum("train hits" + str(train_y)))
