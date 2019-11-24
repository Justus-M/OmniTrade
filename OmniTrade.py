import pandas as pd
import os
import TensorPrep
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
from tfData import tfData
from bayes_opt import BayesianOptimization

def DataPrep(p):

    Data = tfData()

    for ticker in p['Tickers']:
        Frame = pd.read_csv('Data/Minute/%s.csv' % ticker, header=0, index_col = 'timestamp', parse_dates=True)
        Frame.columns = ticker + ' ' + Frame.columns.values
        Frame.index = pd.to_datetime(Frame.index, unit='s')
        Frame = Frame.resample(p['hindsight_interval']).mean()
        try:
            Data.TrainingFrame = Data.TrainingFrame.merge(Frame, how='inner', left_index=True, right_index=True)
        except:
            Data.TrainingFrame = Frame

    Data.TrainingFrame = Data.TrainingFrame[::-1]
    Data.TrainingFrame = Data.TrainingFrame[Data.TrainingFrame.index.year>p['YearCutoff']]
    Data.Split()
    Data.TrainingFrame = TensorPrep.TensorDatasetPreparation(p, Data.TrainingFrame)
    Data.ValidationFrame = TensorPrep.TensorDatasetPreparation(p, Data.ValidationFrame)

    kill = (Data.TrainingFrame['none'] == 0) & (Data.TrainingFrame['none'] == Data.TrainingFrame['none'].shift(periods=-1))
    Data.TrainingFrame['none'] += kill.astype(int)*2

    Data.TsTensorDataset(LabelCount=p['LabelCount'], Exclude=(p['LabelCount']-1)*2, WindowSize=p['hindsight'], Batchsize=p['Batch_size'])

    return Data

def Train(Data = False):

    from Parameters import p
    if Data == False:
        Data = DataPrep(p)

        with open('models/Optimal Hyperparameters.txt') as Hp:
            Hp = Hp.read()
            Hp = eval(Hp)
    else:
        Hp = p['Hp']

    p = DefineLayers(p, Hp)

    Model, history = TrainModel(p, Data)
    Model.evaluate(Data.TfValidationDataset)

    return Model, history

def OptimizationEvaluation(**Parameters):

    params['Hp'] = {}

    Data = DataPull(p)

    for Parameter, Value in Parameters.items():
        params['Hp'][Parameter] = Value

    Model, history = Train(params, Data = Data)

    return history.history['val_accuracy'][-1]

def BayesOptimization(p):

    global params
    params = p

    pbounds = {'MainLSTMlayers': (2, 8),
           'MainLSTMNodes': (32, 256),
           'MainLSTMDropout': (0.15, 0.4),
           'FinalLSTMNodes': (32, 256),
           'FinalLSTMDropout': (0.15, 0.4),
           'FinalDenseNodes': (32, 256),
           'FinalDenseDropout': (0.15, 0.4),
           'LearningRate': (0.0001, 0.01),
           'Decay': (0.00000001, 0.01)
          }

    optimizer = BayesianOptimization(
        f=OptimizationEvaluation,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    optimizer.maximize(init_points=10, n_iter=100)
    print(optimizer.max)

def DefineLayers(p, Hp):

    p['layers'] = [[]] * int(Hp['MainLSTMlayers'])
    for layer in range(0, int(Hp['MainLSTMlayers'])):
        p['layers'][layer].append([int(Hp['MainLSTMNodes']), int(Hp['MainLSTMDropout'])])
    p['Hp'] = Hp

    return p

def SaveModel(p, Model):

    Specs = {}
    Specs['TargetTickers'] = str(p['TargetTickers'])
    Specs['Tickers'] = p['Tickers']
    Specs['nTickers'] = len(p['tickers'])
    Specs['Foresight'] = p['foresight']
    Specs['Hindsight'] = p['hindsight']
    Specs['Interval'] = p['hindsight_interval']

    if not os.path.exists('Models/ModelIndex.csv'):
        Specs['ID'] = 1
        ModelIndex = pd.DataFrame(Specs, index = [1])
    else:
        ModelIndex = pd.read_csv('Models/ModelIndex.csv')
        Specs['ID'] = max(ModelIndex['ID'])+1
        Specs = pd.DataFrame(Specs)
        ModelIndex = ModelIndex.append(Specs)

    Model.save('models/%s.h5' %(str(Specs['ID'])))
    ModelIndex.to_csv('Models/ModelIndex.csv')

def Predict(p, Tensor):

    if not os.path.exists('Models/ModelIndex.csv'):
        print('No Models saved. Train model first.')
        return

    ModelIndex = pd.read_csv('ModelIndex.csv')

    Predictions = Model.predict(Tensor)

    if p['Purpose'] == 'LivePrediction':
        Predictions = pd.DataFrame(Predictions, index=Data['DFrame'].index)
    else:
        Predictions = pd.DataFrame(Predictions, index=p['Data']['TestFrame'].index)
    headers = [a + b for at, b in zip(p['TargetTickers'], ([' long'] * len(p['TargetTickers'])))]
    headers.append('none')
    Predictions.columns = headers

    return Predictions

def TrainModel(p, Data):

    Model = Sequential()

    for layer in p['layers'][0]:
        Model.add(LSTM(layer[0], activation = 'tanh', return_sequences = True))
        Model.add(Dropout(layer[1]))
        Model.add(BatchNormalization())


    Model.add(LSTM(int(p['Hp']['FinalLSTMNodes']), activation = 'tanh'))
    Model.add(Dropout(p['Hp']['FinalLSTMDropout']))
    Model.add(BatchNormalization()) 

    Model.add(Dense(int(p['Hp']['FinalDenseNodes']), activation='relu'))
    Model.add(Dropout(p['Hp']['FinalDenseDropout']))

    Model.add(Dense(p['LabelCount'], activation=p['activation']))

    opt = tf.keras.optimizers.Adam(lr=p['Hp']['LearningRate'] , decay=p['Hp']['Decay'])

    Model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # NAME = f'{p["TargetTickers"][0]}PRED-{p["Hp"]["MainLSTMlayers"]}-MainLayers-{p["Hp"]["MainLSTMNodes"]}-MainNodes-' \
    #     f'{p["Hp"]["FinalLSTMNodes"]}-FinalLSTMNodes-{p["Hp"]["FinalLSTMDropout"]}-FinalLSTMDropout-{p["Hp"]["FinalDenseNodes"]}' \
    #     f'-FinalDenseNodes-{p["Hp"]["FinalDenseDropout"]}-FinalDenseDropout-{p["Hp"]["LearningRate"]}-LearningRate-{p["Hp"]["Decay"]}-Decay'

    # tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


    if p['Purpose'] == 'Training':
        history = Model.fit(Data.TfTrainingDataset, epochs=p['epochs'], validation_data=Data.TfValidationDataset) # , callbacks=[tensorboard]
    else:
        history = Model.fit(Data.TfTrainingDataset, epochs=p['epochs'])

    return Model, history