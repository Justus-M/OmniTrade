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
import os
import numpy as np
import plaidml

importlib.reload(TsDataProcessor)

#modifiable parameters
TargetTicker = "BTC"
epochs = 10
Batch_size = 256
TestProportion = 0.2
ValidationProportion = 0.05
hindsight = 512
foresight = 128
buy_threshold= 0.02
y_name = "close"
x_names = ["close", "volume", "low", "high", "open"]

cryptos = dict.fromkeys(["BTC", "ETH", "LTC", "EOS"])

for i in cryptos:
    cryptos[i] = pd.read_csv("Data/" + i + "-USD-train.csv", names=["time", "open", "close", "high", "low", "volume"],index_col = "time", parse_dates=True)
    cryptos[i] = cryptos[i][x_names] #only keep relevant columns
    cryptos[i].index = pd.to_datetime(cryptos[i].index, unit = 'ms')
    cryptos[i].columns = i + " " + cryptos[i].columns.values #add crypto name to column headers for later when the datasets are combined
    cryptos[i].drop_duplicates(inplace=True)


#the below lines calculate the return until the target date determined by "foresight" in the above parameters, and converts it to a 1 or 0 depending on whether
#or not it is above the buy threshold in the above parameters. Ex. if buy threshold is 0.05 then "target" will be one for a return of 5% or more
cryptos[TargetTicker]["target"] = cryptos[TargetTicker][TargetTicker + " " + y_name].shift(periods=-foresight) / cryptos[TargetTicker][TargetTicker + " " + y_name]
cryptos[TargetTicker]["target"] = (cryptos[TargetTicker]["target"] > 1 + buy_threshold).astype(int) #convert to
cryptos[TargetTicker].dropna(inplace=True)
#
trades = pd.read_csv("Data/" + TargetTicker + "-USD-trades.csv", names=["ID", "time", "AMOUNT", "PRICE"],index_col = "time", parse_dates=True)
trades.index = pd.to_datetime(trades.index, unit = 'ms')
trades = trades.resample("1T").sum()
cryptos[TargetTicker] = cryptos[TargetTicker].merge(trades["AMOUNT"], how ='inner', left_index=True, right_index=True)
cryptos[TargetTicker].dropna(inplace=True)
del trades

TargetPrice = pd.DataFrame(cryptos[TargetTicker][TargetTicker + " " + y_name].shift(periods=-foresight)).copy()
Price = pd.DataFrame(cryptos[TargetTicker][TargetTicker + " " + y_name]).copy()

TargetPrice.columns.values[0] = "TargetPrice"
Price.columns.values[0] = "Price"


#this loop scales the data and puts the data of the non-target currencies together in a frame, for the TsDataProcessor function

altcrypto = pd.DataFrame(index = cryptos[TargetTicker].index)

for i in cryptos:
    cryptos[i] = TsDataProcessor.Scaler(cryptos[i], "target")
    if i != TargetTicker:
        altcrypto = cryptos[i].merge(altcrypto, how ='inner', left_index=True, right_index=True)

#Returns all the input frames together in one frame as "combined", and an array of training data. see TsDataProcessor for more
DFrame, AllSequential = TsDataProcessor.TsDataProcessor(cryptos[TargetTicker], altcrypto, target ="target", t=hindsight)

del cryptos
del altcrypto
#separates the data into validation and training sets, Valpercent is the proportion of the total data used for validation

DFrame = DFrame.iloc[:-hindsight+1]

DFrame = DFrame.merge(Price, how ='inner', left_index=True, right_index=True)
DFrame = DFrame.merge(TargetPrice, how ='inner', left_index=True, right_index=True)

days = sorted(DFrame.index.values)
DayMarker = days[-int(TestProportion * len(days))]

marker = len(DFrame.index.values) - len(DFrame[DFrame.index >= DayMarker])
TestFrame = DFrame.iloc[marker:]
TestSequential = AllSequential[marker:]

TrainFrame = DFrame.iloc[:marker]
TrainSequential = AllSequential[:marker]
TrainFrame.drop(["Price", "TargetPrice"], axis = 1)

del DFrame
del AllSequential

days = sorted(TrainFrame.index.values)

DayMarker = days[-int(ValidationProportion * len(days))]

random.shuffle(TrainSequential)
marker = TrainFrame[TrainFrame.index >= DayMarker]


validation = TrainSequential[:len(marker)]
train = TrainSequential[len(marker):]

#the data is balanced so we have an equal number of buys and sells, otherwise the algorithm
# can get promising results just by never buying (always predicting 0), or always buying (always predicting 1) depending on the data
train = TsDataProcessor.Balance(train)
validation = TsDataProcessor.Balance(validation)

train_x, train_y = TsDataProcessor.split(train)
val_x, val_y = TsDataProcessor.split(validation)
test_x, test_y = TsDataProcessor.split(TestSequential)

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
Model = Sequential()

Model.add(LSTM(128, input_shape =(train_x.shape[1:]), activation = "tanh", return_sequences = True))
Model.add(Dropout(0.3))
Model.add(BatchNormalization())

Model.add(LSTM(128, input_shape =(train_x.shape[1:]), activation = "tanh", return_sequences = True))
Model.add(Dropout(0.3))
Model.add(BatchNormalization())

Model.add(LSTM(128, input_shape =(train_x.shape[1:]), activation = "tanh"))
Model.add(Dropout(0.3))
Model.add(BatchNormalization())

Model.add(Dense(32, activation="relu"))
Model.add(Dropout(0.2))

Model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001 , decay=1e-6)

Model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# NAME = f"{predict}-{hindsight}-SEQ-{foresight}-PRED-{int(time.time())}"
#
# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#
# filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
#
# checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = Model.fit(train_x, train_y, batch_size = Batch_size,epochs = epochs, validation_data=(val_x, val_y))

# Model.evaluate(test_x, test_y)

# TestPredictions = Model.predict(test_x)
# TestFrame["Predictions"] = TestPredictions
# TestFrame.to_csv(r'TestData-foresight:%s Hindsight:%s Buy-Threshold:%s.csv'%
#                  (foresight, hindsight, buy_threshold))

Investment = 1000

for i in range(0,len(Prediction)):
    if Prediction[i] == 1:
        Investment += ((TestFrame["TargetPrice"].iloc[i]-TestFrame["Price"].iloc[i])/TestFrame["Price"].iloc[i])*100
    if Investment < 0:
        print("Bust after " + str(i) + " minutes.")
        break

if Investment > 0:
    print("You made " + str(int(Investment-1000)) + " euros. " + str(int((Investment-1000)/10)) + "% Return.")
