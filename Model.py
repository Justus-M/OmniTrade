from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import TensorPrep
import os

def TrainModel(p, Data):

    os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    Model = Sequential()

    for layer in p['layers'][0]:
        Model.add(LSTM(layer[0], activation = 'tanh', return_sequences = True))
        Model.add(Dropout(layer[1]))
        Model.add(BatchNormalization())


    Model.add(LSTM(p['Hp']['FinalLSTMNodes'], activation = 'tanh'))
    Model.add(Dropout(p['Hp']['FinalLSTMDropout']))
    Model.add(BatchNormalization())

    Model.add(Dense(p['Hp']['FinalDenseNodes'], activation='relu'))
    Model.add(Dropout(p['Hp']['FinalDenseDropout']))

    Model.add(Dense(p['LabelCount'], activation=p['activation']))

    opt = tf.keras.optimizers.Adam(lr=p['Hp']['LearningRate'] , decay=p['Hp']['Decay'])

    Model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    NAME = f'{p["TargetTickers"][0]}PRED-{p["Hp"]["MainLSTMlayers"]}-MainLayers-{p["Hp"]["MainLSTMNodes"]}-MainNodes-' \
        f'{p["Hp"]["FinalLSTMNodes"]}-FinalLSTMNodes-{p["Hp"]["FinalLSTMDropout"]}-FinalLSTMDropout-{p["Hp"]["FinalDenseNodes"]}' \
        f'-FinalDenseNodes-{p["Hp"]["FinalDenseDropout"]}-FinalDenseDropout-{p["Hp"]["LearningRate"]}-LearningRate-{p["Hp"]["Decay"]}-Decay'

    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
    # filepath = f'RNN_Final-{epoch:02d}'  # unique file name that will include the epoch and the validation acc for that epoch
    # filepath = f'RNN_Final-{epoch:02d}-{val_acc:.3f}'  # unique file name that will include the epoch and the validation acc for that epoch

    # checkpoint = ModelCheckpoint('models/{}.model'.format(filepath, verbose=1, save_best_only=True, mode='max')) # saves only the best ones

    if p['Purpose'] == 'Training':
        history = Model.fit(Data['Tensor'], epochs=p['epochs'], validation_data=Data['ValTensor'], callbacks=[tensorboard])
    else:
        history = Model.fit(Data['Tensor'], epochs=p['epochs'])


    # Model.evaluate(test_x, test_y)

    return Model, history