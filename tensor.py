import timeimport pandas as pdimport importlibimport TsDataProcessorfrom tensorflow.keras.models import Sequentialfrom tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalizationfrom tensorflow.keras.callbacks import TensorBoard, ModelCheckpointimport tensorflow as tfimport randomimport Importsimport torchimport osimport numpy as np#import plaidmlimportlib.reload(Imports)importlib.reload(TsDataProcessor)#modifiable parametersTargetTickers = ["BTC"]epochs = 5Batch_size = 256TestProportion = 0.4ValidationProportion = 0.1hindsight = 64hindsight_interval= "10T"foresight = 16foresight_interval = hindsight_intervalbuy_threshold= 0.02y_name = "close"x_names = ["close", "volume", "low", "high", "open"]target, TargetPrice, Price = Imports.ImportCrypto(x_names, y_name, TargetTickers, foresight, hindsight, buy_threshold, ["BTC", "EOS", "ETH"], hindsight_interval)DFrame = TsDataProcessor.TsDataProcessor(target, target ="target", t=hindsight)del targettestlength = int(TestProportion*len(DFrame))test = DFrame.iloc[:testlength]DFrame = DFrame.iloc[testlength:]tf.enable_eager_execution()def Tensify (dataframe, hindsight, Y = True):    if Y:        variables = (tf.constant(dataframe[dataframe.columns.values[:-1]].values), tf.constant(dataframe[dataframe.columns.values[-1]].values))    else:        variables = (tf.constant(dataframe.values))    tensor = tf.data.Dataset.from_tensor_slices(variables)    tensor = tensor.window(hindsight,1,1,True)    if Y:        tensor = tensor.flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(hindsight), y.batch(1))))    else:        tensor = tensor.flat_map(lambda x: tf.data.Dataset.zip(x.batch(hindsight)))    return tensordef BalanceTensor(tensor, npos):    positive = tensor.filter(lambda x,y: tf.math.equal(y[0],1))    negative = tensor.filter(lambda x,y: tf.math.equal(y[0],0))    negative = negative.shuffle(20000)    negative = negative.take(npos)    tensor = positive.concatenate(negative)    tensor = tensor.shuffle(20000)    return tensortensor = Tensify(DFrame, hindsight)tensor = BalanceTensor(tensor, sum(DFrame["target"]))tensor = tensor.batch(Batch_size).prefetch(1)test = test.merge(Price, how ='inner', left_index=True, right_index=True)test = test.merge(TargetPrice, how ='inner', left_index=True, right_index=True)TestTensor = Tensify(test[test.columns.values[:-3]], hindsight, Y = False)TestTensor = TestTensor.batch(Batch_size).prefetch(1)test = test.iloc[:-hindsight+1]del DFrameos.environ["KERAS_BACKEND"] = "plaidml.keras.backend"Model = Sequential()Model.add(LSTM(128, activation = "tanh", return_sequences = True))Model.add(Dropout(0.3))Model.add(BatchNormalization())Model.add(LSTM(128, activation = "tanh", return_sequences = True))Model.add(Dropout(0.3))Model.add(BatchNormalization())Model.add(LSTM(128, activation = "tanh"))Model.add(Dropout(0.3))Model.add(BatchNormalization())Model.add(Dense(32, activation="relu"))Model.add(Dropout(0.2))Model.add(Dense(2, activation="softmax"))opt = tf.keras.optimizers.Adam(lr=0.001 , decay=1e-6)Model.compile(loss="sparse_categorical_crossentropy",              optimizer=opt,              metrics=["accuracy"])# NAME = f"{predict}-{hindsight}-SEQ-{foresight}-PRED-{int(time.time())}"## tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))## filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch## checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best oneshistory = Model.fit(tensor, epochs = epochs)# Model.evaluate(test_x, test_y)TestPredictions = Model.predict(TestTensor)# TestFrame["Prediction"] = PredictionPrediction = []for i in range(0, len(TestPredictions)):    if TestPredictions[i][1]>TestPredictions[i][0]:        Prediction.append(1)    else:        Prediction.append(0)Investment = 1000count = 0for i in range(len(Prediction)-1,-1,-1):    if Prediction[i] == 1:        Investment += ((test[TargetTickers[0] + " TargetPrice"].iloc[i]-test[TargetTickers[0] + " Price"].iloc[i])/test[TargetTickers[0] + " Price"].iloc[i])*10        if (test[TargetTickers[0] + " TargetPrice"].iloc[i]-test[TargetTickers[0] + " Price"].iloc[i]) > 0:            count+= 1    if Investment < 0:        print("Bust after " + str(i) + " minutes.")        breakif Investment > 0:    print("You made " + str(int(Investment-1000)) + " euros. " + str(int((Investment-1000)/10)) + "% Return in " + str(int(len(Prediction)/6/24)) + " days.")    print(str(sum(Prediction)) + " trades with a " + str(int(count*100/sum(Prediction))) + "% success rate.")