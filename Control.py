import tensorflow as tf
import ModelTraining
import TradeSimulation
from Parameters import p
import Prediction

p['TargetTickers'] = ['MSFT']
p['foresight'] = 16
p['Purpose'] = 'LivePrediction'

p['Data'] = ModelTraining.DataPull(p)
Prediction = Prediction.Predict(p, p['Data']['Tensor'])