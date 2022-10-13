import pandas as pd
import os, sys, time, logging
from featureengineering import feature_engineering, feature_engineering_stack
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import tensorflow as tf
from tfclasses import TfData
import dataupdate
import numpy as np
from importlib import reload
from sklearn.model_selection import KFold
import sys
sys.path.insert(1, '/Users/justusmulli/projects/mltoolkit')
import kerastuner as kt
import utils
import pyarrow.parquet as pq
from Helpers import *
from utils import *
from params import *
from time_series_data_handler import *


#aws s3 sync s3://omni-raw-data/ /users/justusmulli/projects/omnitrade/awsfj

def get_data_for_prediction(params: Params, omni_tf_data: TfData):
    omni_tf_data.rename_pred()
    omni_tf_data.pred_frame = omni_tf_data.pred_frame.iloc[:params.hindsight]
    omni_tf_data.pred_frame = feature_engineering(params, omni_tf_data.pred_frame, realized_predictions=False)
    omni_tf_data.time_series_tf_dataset(label_count=0, window_size=params.hindsight,
                                        batch_size=params.batch_size)
    return omni_tf_data

def cross_validation(model_builder, argument, folds, omni_parameters):
    cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='max', patience=10,
                                          restore_best_weights=True)
    metric, y, ypred = [], [], []
    for i in range(folds):
        model = model_builder(argument)
        training_data = data_prep(omni_parameters, fold=[i, folds])
        for file in os.listdir():
            if 'status' in file:
                os.remove(file)
        split = file.split(" fold ")[0]
        open(f'{split} fold {i} of {folds}', 'w').close()
        history = model.fit(training_data.tf_training_dataset, epochs=100   ,
                            validation_data=training_data.tf_validation_dataset, use_multiprocessing=True,
                            workers=16, callbacks=[cb], class_weight=training_data.weights)

        metric.append(min(history.history['val_loss']))
        print(f'fold {i + 1}')
        y.extend(np.concatenate([y for x, y in training_data.tf_validation_dataset], axis=0))
        ypred.extend(history.model.predict(training_data.tf_validation_dataset))
    heatmapper(ypred, y)
    print(f'mean metric {np.mean(metric)}')
    return np.mean(metric), history, y, ypred

def cv_bayesian_tuning(builder, omni_params, objective=kt.Objective('val_loss', 'max'), max_trials=100):

    class bayes(kt.tuners.bayesian.BayesianOptimization):

        def run_trial(self, trial):
            try:
                self.trial_times.append(time.time)
            except:
                self.trial_times = [time.time()]
            mean_metric, history, y, ypred = cross_validation(self.hypermodel.build, trial.hyperparameters, 5, omni_params)
            self.oracle.update_trial(trial.trial_id, {'val_loss': mean_metric})
            self.save_model(trial.trial_id, history.model)
            try:
                self.n +=1
            except:
                self.n = 1
            for file in os.listdir():
                if 'status' in file:
                    os.remove(file)
            self.trial_times[-1] = time.time()-self.trial_times[-1]
            open(f'status - step {self._reported_step} of {self.remaining_trials} - time - {np.mean(self.trial_times)}', 'w').close()

    return bayes(builder, objective, max_trials, num_initial_points=10)

def train(omni_params = False, folds = False, search = False, hyperparams = None):
    if not hyperparams:
        hyperparams = kt.HyperParameters()
    if not omni_params:
        omni_params = reload_params()

    cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'max', patience=10,
                                          restore_best_weights=True)

    tuner = cv_bayesian_tuning(builder, omni_params)

    if not search:
        if not folds:
            model = tuner.hypermodel.build(hyperparams)
            training_data = data_prep(omni_params)
            history = model.fit(training_data.tf_training_dataset, epochs=omni_params['epochs'],
                                validation_data=training_data.tf_validation_dataset, use_multiprocessing=True,
                                workers=16, callbacks=[cb],class_weight=training_data.weights)
        else:
            mean_metric, history, y, ypred = cross_validation(tuner.hypermodel.build, hyperparams, 5, omni_params)
    else:
        tuner.search()
        metric, hist, y, ypred = cross_validation(tuner.hypermodel.build, tuner.get_best_hyperparameters()[0], 5, omni_params)
        set_specs(omni_params)
        utils.pickle_save({'params': tuner.get_best_hyperparameters()[0], 'y': y, 'ypred':ypred}, ''.join([omni_params['specs'][p] for p in omni_params['specs'].keys()]))
        history = tuner

    return history, y, ypred

def set_specs(params):
    params['specs'] = {}
    for key in ['foresight', 'hindsight', 'buy_threshold', 'sell_threshold', 'hindsight_interval']:
        params['specs'][key] = str(params[key])
    params['specs']['n_tickers'] = str(len(params['tickers']))
    params['specs']['n_target_tickers'] = str(len(params['target_tickers']))

def results(params = None):
    if params == None:
        params = reload_params()

    res = utils.pickle_load(''.join([params['specs'][p] for p in params['specs'].keys()]))
    heatmapper(res['ypred'], res['y'])

def save_model(params, Model, hist):

    model_specs_dict = params['specs']
    model_specs_dict['validation_accuracy'] = hist.history['val_accuracy'][-1]


    if not os.path.exists('models/Model Index.csv'):
        model_specs_dict['ID'] = 1
        model_index = pd.DataFrame.from_dict(model_specs_dict)
        model_index.set_index('ID', inplace = True)
    else:
        model_index = pd.read_csv('Models/Model Index.csv', index_col= 'ID')
        model_specs_dict['ID'] = model_index.index[-1] + 1
        model_specs_frame = pd.DataFrame(model_specs_dict)
        model_specs_frame.set_index('ID', inplace = True)
        if model_specs_frame.values in model_index.values:
            filter = model_index.index>0
            for col in model_specs_frame.columns.values:
                filter &= model_index[col] == model_specs_frame[col]
            if model_index[filter]['validation_accuracy'] < model_specs_frame['validation_accuracy']:
                model_specs_frame.index[0] = model_index.index[filter][0]
                model_index.drop(model_specs_frame['ID'], inplace = True)
                acc = int(model_specs_dict['validation_accuracy'] * 100)
                print(f'New max model accuracy of {acc}% for this specification. Saving model.')
            else:
                return

        model_index = model_index.append(model_specs_frame)
        model_index.sort_index(inplace = True)

    Model.save(f'Models/{str(params["specs"]["ID"])}.h5')
    model_index.to_csv('Models/Model Index.csv')

    with open('models/Optimal Hyperparameters.txt', 'w') as opt_hyperparams:
        opt_hyperparams.write(str(params['hyperparams']))

def predict(input_data, omni_params = None):

    if omni_params is None:
        omni_params = reload_params()

    def no_model():
        print('No saved models with matching specifications. Train model first.')
        sys.exit(0)

    if not os.path.exists('Models/Model Index.csv'):
        no_model()

    model_index = pd.read_csv('models/Model Index.csv', index_col= 'ID')

    filter = model_index.index > 0
    for specification in omni_params['specs']:
        filter &= omni_params['specs'][specification] == model_index[specification]

    if sum(filter)==0:
        no_model()

    model_ID = model_index.index[filter][0]

    model = load_model('Models/%s.h5' % model_ID)

    predictions = model.predict(input_data.tf_pred_dataset)

    if omni_params['purpose'] == 'live_prediction':
        predictions = pd.DataFrame(predictions, index=[input_data.pred_frame.index.values[0]])
    else:
        predictions = pd.DataFrame(predictions, index=input_data.test_frame.index)
    headers = []
    for ticker in omni_params['target_tickers']:
        headers.append(ticker + ' long')
        if omni_params['sell_threshold'] != None:
            headers.append(ticker + ' short')
    headers.append('none')
    predictions.columns = headers

    return predictions

def live_feed():

    omni_params = reload_params()

    omni_params['purpose'] = 'live_prediction'
    interval = omni_params['hindsight_interval'].replace('T','')

    suppress_tf_warnings()
    
    print(f'Printing recommendations every {interval} minutes.')

    while True:

        last_open, EOD, now = dataupdate.last_market_open()
        dataupdate.alphavantage_update(omni_params['tickers'])

        latest_data = data_prep(omni_params)

        prediction = predict(latest_data, omni_params)

        print(prediction)

        signal = prediction.values[0]
        if max(signal) != signal[-1]:

            max_index = np.where(signal == max(signal))[0][0]
            max_header = prediction.columns.values[max_index].split()
            position = max_header[1]
            ticker = max_header[0]

            print(f'Recommendation to go {position} on {ticker} stock.')
        else:
            print('Recommendation not to enter any new positions')


        if last_open != now:
            print("Market Closed")
            break
        time.sleep(60*int(interval))

def builder(hp):
    model = Sequential()
    params = reload_params()

    for layer in range(hp.Int('layers', 0, 5, default = 1)):
        model.add(LSTM(hp.Int(f'units_{layer}', 4, 128, step = 4, default = 8), activation='tanh', return_sequences=True))
        model.add(Dropout(hp.Float(f'dropout_{layer}', 0, 0.9, default = 0.4)))
        if hp.Boolean(f'batch_norm_{layer}', default = True):
            model.add(BatchNormalization())

    model.add(LSTM(hp.Int('Final_LSTM_Nodes', 4, 128, step = 4, default = 8), activation='tanh'))
    model.add(Dropout(hp.Float('Final_LSTM_dropout', 0, 0.9, default = 0.4)))
    if hp.Boolean('Final_LSTM_batch_norm', default = True):
        model.add(BatchNormalization())

    model.add(Dense(hp.Int('Final_Dense_Nodes', 4, 128, step = 4, default = 8), activation='relu'))
    model.add(Dropout(hp.Float('Final_Dense_dropout', 0, 0.9, default = 0.4)))

    model.add(Dense(params['label_count'], activation=params['activation']))

    opt = tf.keras.optimizers.Adam(lr=hp.Float('lr', 0.0000001, 0.1, default = 0.005), decay=hp.Float('decay', 0.000000001, 0.005, default = 0.000005))
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(), tf.keras.metrics.SensitivityAtSpecificity(0.6), tf.keras.metrics.SpecificityAtSensitivity(0.6)]
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=metrics)
    return model

def suppress_tf_warnings():
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def reload_params():
    import params
    reload(params)
    return params.omni_params