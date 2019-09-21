import tensorflow as tf
import ModelTraining
import TradeSimulation
from Parameters import p

p['TargetTickers'] = ['MSFT']
p['foresight'] = 16
p['Purpose'] = 'TestPrediction'

Data = ModelTraining.DataPull(p)
Model, history = ModelTraining.Train(p)


TestPredictions = Model.predict(Data['TestTensor'])
TradeSimulation.simulate(p, TestPredictions, Data['TestFrame'])
