from DataPrep import DataPreparation
from Model import TrainModel
import pandas as pd
import os
import TradeSimulation
import TensorPrep
import tensorflow as tf
import itertools
import numpy as np
import pickle
from tensorflow.keras.models import load_model

def DataPull(p):

    if not os.path.exists('Data/Processed%s.csv' % p['hindsight_interval']):
        print('Running DataPreparation as %s frequency prepared data was not found' % p['hindsight_interval'])
        DataPreparation(p)

    Data = {}
    Data['DFrame'] = pd.read_csv('Data/Processed%s.csv' % (p['hindsight_interval']), index_col = 'timestamp', parse_dates=True)
    Data['DFrame'] = Data['DFrame'][75000:][::-1]

    if p['Purpose'] == 'TestPrediction':
        Data['DFrame'], Data['TestFrame'] = TensorPrep.Split(Data['DFrame'], p['TestProportion'])
        Data['TestTensor'], Data['TestFrame'] = TensorPrep.TensorPreparation(p, Data['TestFrame'], Balance=False)
        Data['Tensor'], Data['DFrame'] = TensorPrep.TensorPreparation(p, Data['DFrame'])
        return Data

    if p['Purpose'] == 'Training':
        Data['DFrame'], Data['ValidationFrame'] = TensorPrep.Split(Data['DFrame'], p['ValidationProportion'])
        Data['ValidationTensor'], Data['ValidationFrame'] = TensorPrep.TensorPreparation(p, Data['ValidationFrame'])
        Data['Tensor'], Data['DFrame'] = TensorPrep.TensorPreparation(p, Data['DFrame'])
        return Data

    elif p['Purpose'] == 'LivePrediction':
        units = len(Data['DFrame'])
        Data['Tensor'], Data['DFrame'] = TensorPrep.TensorPreparation(p, Data['DFrame'].iloc[:p['hindsight']],
                                                                      Balance=False, RealizedPredictions=False)
        return Data
    else:
        print('Purpose must be LivePrediction, TestPrediction, or Training')
        exit()

def DefineLayers(p, Hp):

    p['layers'] = [[]] * Hp['MainLSTMlayers']
    for layer in range(0, Hp['MainLSTMlayers']):
        p['layers'][layer].append([Hp['MainLSTMNodes'], Hp['MainLSTMDropout']])
    p['Hp'] = Hp

    return p

def HyperparameterSearch(p):

    Data = DataPull(p)

    Hyperparameters = pd.read_csv('models/Hyperparameters.csv')

    Hp = {}
    for header in Hyperparameters.columns.values:
        Hp[header] = Hyperparameters[header].dropna()

    Combinations = list(itertools.product(*list(Hp.values())))

    headers = list(Hp)
    headers.extend(['Val_acc', 'Training_acc'])

    if not os.path.exists('models/Model-trials.csv'):
        temp = pd.DataFrame(columns=headers, index = [1])
        temp.to_csv('models/Model-trials.csv')

    Models = pd.read_csv('models/Model-trials.csv', index_col = 0)

    for Hpcombination in Combinations:
        index = ''.join(list(str(Hpcombination)))
        if index not in Models.index:
            add = pd.DataFrame([Hpcombination], columns = list(Hp), index = [index])
            add['Training_acc']=False
            add['Val_acc']=False
            Models = Models.append(add)

            for i in range(0,len(Hpcombination)):
                Hp[Hyperparameters.columns.values[i]] = Hpcombination[i]

            p = DefineLayers(p, Hp)

            print(str(len(Combinations)-len(Models)) + ' models left to test. ' + str(len(Models)) + ' tested. ' + str(len(Models)*100/len(Combinations)) + ' %.')

            Model, history = TrainModel(p, Data)
            Models['Val_acc'].loc[index] = history.history['val_accuracy'][-1]
            Models['Training_acc'].loc[index] = history.history['accuracy'][-1]
            Models.dropna(inplace = True)
            Models[headers].to_csv('models/Model-trials.csv')
            if Models['Val_acc'].loc[index] == max(Models['Val_acc']):
                Model.save('models/%s-%d-foresight.h5' %(str(p['TargetTickers']), p['foresight']))
                Optimal = open('models/Optimal Hyperparameters.txt', 'w')
                Optimal.write(str(Hp))
                Optimal.close()

def Train(p):

    Data = DataPull(p)


    Models = pd.read_csv('models/Model-trials.csv', index_col = 0)
    Models.dropna(inplace = True)

    with open('models/Optimal Hyperparameters.txt') as Hp:
        Hp = Hp.read()
        Hp = eval(Hp)

    p = DefineLayers(p, Hp)

    Model, history = TrainModel(p, Data)

    return Model, history

def SaveModel(p, Model):

    Specs = {}
    Specs['TargetTickers'] = str(p['TargetTickers'])
    Specs['nTickers'] = len(p['tickers'])
    Specs['Foresight'] = p['foresight']
    Specs['Findsight'] = p['hindsight']
    Specs['Interval'] = p['foresight_interval']
    Specs['HindsightExtension'] = str(p['HindsightExtension'])
    Specs['HyperParamsID'] = p['HyperParamsID']

    if not os.path.exists('Models/ModelIndex.csv'):
        Specs['ID'] = 1
        ModelIndex = pd.DataFrame(Specs, index = 1)
    else:
        ModelIndex = pd.read_csv('Models/ModelIndex.csv')
        Specs['ID'] = max(ModelIndex['ID'])+1
        Specs = pd.DataFrame(Specs)
        ModelIndex = ModelIndex.append(Specs)

    Model.save('models/&s.h5' %(str(Specs['ID'])))
    ModelIndex.to_csv('Models/ModelIndex.csv')

def Predict(p, Tensor):

    p['TargetTickers'].sort()

    if os.path.exists('models/%s-%d-foresight.h5' %(str(p['TargetTickers']), p['foresight'])):
        print('Loading pre-trained model.')
        Model = load_model('models/%s-%d-foresight.h5' %(str(p['TargetTickers']), p['foresight']))
    else:
        print('No pre-trained model matching requirements. Train model for prediction?')
        answer = None
        while answer not in ('y', 'n'):
            answer = input('y/n: ')
            if answer == 'y':
                Model, history = Train(p)
                SaveModel(p, Model)
            elif answer == 'n':
                return
            else:
                print("Please enter y or n.")

    Predictions = Model.predict(Tensor)
    if p['Purpose'] == 'LivePrediction':
        Predictions = pd.DataFrame(Predictions, index=p['Data']['DFrame'].index)
    else:
        Predictions = pd.DataFrame(Predictions, index=p['Data']['TestFrame'].index)
    headers = [a + b for a, b in zip(p['TargetTickers'], ([' long'] * len(p['TargetTickers'])))]
    headers.append('none')
    Predictions.columns = headers

    return Predictions
