from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import DataTransformation
import os

class Parameters:
    pass

p = Parameters()

p.TargetTickers = ['BTC', 'EOS']
p.epochs = 5
p.Batch_size = 256
p.TestProportion = 0.4
p.ValidationProportion = 0.1
p.hindsight = 512
p.hindsight_interval= "1T"
p.foresight = 128
p.foresight_interval = p.hindsight_interval
p.buy_threshold= 0.02
p.sell_threshold= 0.05
p.y_name = "close"
p.x_names = ["close", "volume", "low", "high", "open"]
p.tickers = ['BTC', 'EOS', 'LTC']

p.displace = 2
if p.sell_threshold == None:
    p.displace = 1

if p.sell_threshold == None and len(p.TargetTickers) == 1:
    p.activation = "softmax"
else:
    p.activation = "sigmoid"

p.LabelCount = len(p.TargetTickers) * p.displace + 1

def TrainModel():

    Tensor, TestTensor, p = DataTransformation.DataPrep(p)
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    Model = Sequential()

    Model.add(LSTM(128, activation = "tanh", return_sequences = True))
    Model.add(Dropout(0.3))
    Model.add(BatchNormalization())

    Model.add(LSTM(128, activation = "tanh", return_sequences = True))
    Model.add(Dropout(0.3))
    Model.add(BatchNormalization())

    Model.add(LSTM(128, activation = "tanh"))
    Model.add(Dropout(0.3))
    Model.add(BatchNormalization())

    Model.add(Dense(32, activation="relu"))
    Model.add(Dropout(0.2))

    Model.add(Dense(p.LabelCount, activation=p.activation))

    opt = tf.keras.optimizers.Adam(lr=0.001 , decay=1e-6)

    Model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    # NAME = f"{predict}-{hindsight}-SEQ-{foresight}-PRED-{int(time.time())}"
    # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    # filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    # checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
    history = Model.fit(Tensor, epochs = p.epochs)

    # Model.evaluate(test_x, test_y)

    TestPredictions = Model.predict(TestTensor)
    return history, TestPredictions, Model