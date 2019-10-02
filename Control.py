import tensorflow as tf
import ModelTraining
import TradeSimulation
from Parameters import p
from matplotlib import pyplot
import time

then = time.time()

p['foresight'] = 16
p['Purpose'] = 'TestPrediction'
p['HyperParamsID'] = 0

p['Data'] = ModelTraining.DataPull(p)
print(time.time()-then)
Prediction = ModelTraining.Predict(p, p['Data']['TestTensor'])
print(time.time()-then)
pyplot.plot(Prediction)
pyplot.legend(Prediction.columns.values)


for i in range(0, len(Prediction.columns.values)):
    if Prediction[Prediction.columns.values[i]].iloc[0] > 0.52:
        print('Recommending ' + Prediction.columns.values[i])

print(time.time()-then)
