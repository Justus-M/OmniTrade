from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras import backend as K
import tensorflow as tf
import keras_tuner as kt
from params import *
import numpy as np
import time
from time_series_data_handler import *
import pandas as pd
import shutil
import os

class MarketForecastNN:

    def __init__(self):
        print('initializing model class')
        self.params = Params()
        self.tuner = bayes(self.build_model, objective='val_loss', max_trials=50)



    def build_model(self,  hp: kt.HyperParameters):
        model = Sequential()

        for layer in range(hp.Int('layers', 0, 2, default=1)):
            model.add(
                LSTM(hp.Int(f'units_{layer}', 4, 24, step=4, default=8), activation='tanh', return_sequences=True))
            model.add(Dropout(hp.Float(f'dropout_{layer}', 0, 0.7, default=0.4)))
            if hp.Boolean(f'batch_norm_{layer}', default=True):
                model.add(BatchNormalization())

        model.add(LSTM(hp.Int('Final_LSTM_Nodes', 4, 24, step=4, default=8), activation='tanh'))
        model.add(Dropout(hp.Float('Final_LSTM_dropout', min_value= 0.1, max_value=0.7, default=0.4)))
        if hp.Boolean('Final_LSTM_batch_norm', default=True):
            model.add(BatchNormalization())

        model.add(Dense(hp.Int('Final_Dense_Nodes', 4, 24, step=4, default=8), activation='relu'))
        model.add(Dropout(hp.Float('Final_Dense_dropout', 0.1, 0.7, default=0)))

        model.add(Dense(self.params.label_count))

        opt = tf.keras.optimizers.Adam(lr=hp.Float('learning_rate', 0.00001, 0.05, default=0.005),
                                       decay=hp.Float('decay', 0.00000001, 0.005, default=0.000005))
        metrics = [tf.keras.metrics.MeanSquaredError()]
        model.compile(loss='mse',
                      optimizer=opt,
                      metrics=metrics)
        print(f'running trial: {hp.values}')


        return model

    def bayesian_search(self, data):
        self.tuner.search()
        self.tuner.get_best_models(1)
        self.model = self.tuner.get_best_models(1)[0]

    def train_model(self, data):
        self.model = self.build_model(kt.HyperParameters())
        self.model.fit(data.training_data, epochs=100,
                            validation_data=data.validation_data, use_multiprocessing=True,
                            workers=16)



class bayes(kt.tuners.bayesian.BayesianOptimization):

    def run_trial(self, trial):
        print(f'running trial: {trial.trial_id}')
        print(f'running trial: {trial.hyperparameters.values}')

        data = TimeSeriesDataHandler(Params())
        try:
            self.trial_times.append(time.time())
        except:
            self.trial_times = [time.time()]

        cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                              restore_best_weights=True)
        history = self.hypermodel.build(trial.hyperparameters).fit(data.training_data, epochs=100,
                            validation_data=data.validation_data, use_multiprocessing=True,
                            workers=16, callbacks = [cb])
        if K.backend() == 'tensorflow':
            K.clear_session()

        self.oracle.update_trial(trial.trial_id, {'val_loss': history.history['val_loss'][-1]})
        pd.read_csv('trials.csv').append({'trial': trial.trial_id, 'params': str(trial.hyperparameters.values), 'mse': history.history['val_loss'][-1]}, ignore_index=True).to_csv('trials.csv', index = False)
        try:
            self.n +=1
        except:
            self.n = 1

        self.trial_times[-1] = time.time()-self.trial_times[-1]
        if os.path.isdir("/content/gdrive/MyDrive"):
            if os.path.isdir("/content/gdrive/MyDrive/untitled_project"):
                shutil.rmtree("/content/gdrive/MyDrive/untitled_project")
            shutil.copytree("untitled_project", "/content/gdrive/MyDrive/untitled_project")
        #open(f'status - step {self._reported_step} of {self.remaining_trials} - time - {np.mean(self.trial_times)}', 'w').close()