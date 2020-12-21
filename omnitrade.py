import pandas as pd
import os, sys, time, logging
from featureengineering import feature_engineering, feature_engineering_stack
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import tensorflow as tf
from tfclasses import tf_data
from bayes_opt import BayesianOptimization
import dataupdate
import numpy as np
from importlib import reload
from sklearn.model_selection import KFold
import sys
import kerastuner as kt
import pyarrow.parquet as pq

#aws s3 sync s3://omni-raw-data/ /users/justusmulli/projects/omnitrade/awsfj

def suppress_tf_warnings():
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def reload_params():
    import params
    reload(params)
    return params.omni_params

def data_prep(omni_params, fold = False):

    omni_tf_data = tf_data()

    for ticker in omni_params['tickers']:
        #raw_frame = pq.ParquetDataset('/users/justusmulli/projects/omnitrade/aws/aapl').read().to_pandas()
        #raw_frame.set_index('time', inplace = True)
        #raw_frame.index = pd.to_datetime(raw_frame.index)
        raw_frame = pd.read_csv(f'{str(omni_params["data_path"])}/{ticker}.csv', header=0, index_col ='timestamp', parse_dates=True)
        raw_frame.columns = ticker + ' ' + raw_frame.columns.values
        raw_frame.index = pd.to_datetime(raw_frame.index, unit='s')
        raw_frame = raw_frame.resample(omni_params['hindsight_interval']).mean()
        raw_frame.dropna(inplace = True)
        try:
            omni_tf_data.training_frame = omni_tf_data.training_frame.merge(raw_frame, how='inner', left_index=True, right_index=True)
        except:
            omni_tf_data.training_frame = raw_frame

    omni_tf_data.training_frame = omni_tf_data.training_frame[::-1]

    if omni_params['purpose'] == 'training':
        omni_tf_data.training_frame = omni_tf_data.training_frame[omni_tf_data.training_frame.index.year > omni_params['year_cutoff']]
        if not fold:
            omni_tf_data.folds = False
            omni_tf_data.split()
        else:
            omni_tf_data.folds = fold
            folder = KFold(n_splits=fold[1])
            folds = list(folder.split(omni_tf_data.training_frame))
            train, val = folds[fold[0]]
            omni_tf_data.validation_frame = omni_tf_data.training_frame.iloc[val].copy()
            omni_tf_data.test_frame = omni_tf_data.training_frame.iloc[val[-1]:].copy()
            omni_tf_data.splits.extend(['test', 'validation'])
            omni_tf_data.training_frame = omni_tf_data.training_frame.iloc[:val[0]]

        if fold:
            if fold[0] in list(range(fold[1]))[1:-1]:
                omni_tf_data.training_frame = feature_engineering(omni_params, omni_tf_data.training_frame)
                omni_tf_data.test_frame = feature_engineering(omni_params, omni_tf_data.test_frame)
            elif not fold[0]:
                omni_tf_data.training_frame = omni_tf_data.test_frame

        omni_tf_data.training_frame = feature_engineering(omni_params, omni_tf_data.training_frame)

        ## Find consecutive buy signals and change classifier so it is filtered out when the classes are balanced. This prevents overfitting.
        consecutive_buys = (omni_tf_data.training_frame['none'] != 1) & (
                omni_tf_data.training_frame['none'].shift(periods=-1) != 1)
        omni_tf_data.training_frame['none'] += consecutive_buys.astype(int) * 2

        try:
            consecutive_buys = (omni_tf_data.test_frame['none'] != 1) & (
                    omni_tf_data.test_frame['none'].shift(periods=-1) != 1)
            omni_tf_data.test_frame['none'] += consecutive_buys.astype(int) * 2
        except:
            pass



        omni_tf_data.validation_frame = feature_engineering(omni_params, omni_tf_data.validation_frame)
        omni_tf_data.time_series_tf_dataset(label_count = omni_params['label_count'], cols_exclude = omni_params['price_count'], window_size = omni_params['hindsight'],
                                   batch_size = omni_params['batch_size'])
        omni_tf_data.weights  = {i: 1 / sum(omni_tf_data.training_frame.values[:,-(omni_params['price_count']+1-(omni_params['label_count']-i))]) for i in range(omni_params['label_count'])}
    elif omni_params['purpose'] == 'live_prediction':
        omni_tf_data.rename_pred()
        omni_tf_data.pred_frame = omni_tf_data.pred_frame.iloc[:omni_params['hindsight']]
        omni_tf_data.pred_frame = feature_engineering(omni_params, omni_tf_data.pred_frame, realized_predictions=False)
        omni_tf_data.time_series_tf_dataset(label_count=0, window_size=omni_params['hindsight'], batch_size=omni_params['batch_size'])

    return omni_tf_data

def cross_validation(model_builder, argument, folds, omni_parameters, cb):
    metric = []
    for i in range(folds):
        model = model_builder(argument)
        training_data = data_prep(omni_parameters, fold=[i, folds])
        history = model.fit(training_data.tf_training_dataset, epochs=100,
                            validation_data=training_data.tf_validation_dataset, use_multiprocessing=True,
                            workers=16, callbacks=[cb], class_weight=training_data.weights)
        metric.append(min(history.history['val_accuracy']))
        print(f'fold {i + 1}')
    print(f'mean metric {np.mean(metric)}')
    return np.mean(metric), history

def train(omni_params = False, folds = False, search = False):

    if not omni_params:
        omni_params = reload_params()

    cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'max', patience=10,
                                          restore_best_weights=True)

    def cv_bayesian_tuning(builder, objective=kt.Objective('val_auc', 'max'), max_trials=100):

        class bayes(kt.tuners.bayesian.BayesianOptimization):

            def run_trial(self, trial):
                mean_metric, history = cross_validation(self.hypermodel.build, trial.hyperparameters, 5, omni_params, cb)
                self.oracle.update_trial(trial.trial_id, {'val_auc': mean_metric})
                self.save_model(trial.trial_id, history.model)

        return bayes(builder, objective, max_trials, num_initial_points=10)

    tuner = cv_bayesian_tuning(builder)

    if not search:
        if not folds:
            model = tuner.hypermodel.build(kt.HyperParameters())
            training_data = data_prep(omni_params)
            history = model.fit(training_data.tf_training_dataset, epochs=omni_params['epochs'],
                                validation_data=training_data.tf_validation_dataset, use_multiprocessing=True,
                                workers=16, callbacks=[cb],class_weight=training_data.weights)
        else:
            mean_metric, history = cross_validation(tuner.hypermodel.build, kt.HyperParameters(), 5, omni_params, cb)
    else:
        tuner.search()
        history = tuner

    return history

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
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=metrics)
    return model

