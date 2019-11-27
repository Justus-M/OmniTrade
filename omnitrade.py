import pandas as pd
import os
from featureengineering import feature_engineering
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
from tfclasses import tf_data
from bayes_opt import BayesianOptimization
import dataupdate
import time
import logging

def suppress_tf_warnings():
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def data_prep(omni_params):

    omni_tf_data = tf_data()

    for ticker in omni_params['tickers']:
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
        omni_tf_data.split()
        omni_tf_data.training_frame = feature_engineering(omni_params, omni_tf_data.training_frame)
        omni_tf_data.validation_frame = feature_engineering(omni_params, omni_tf_data.validation_frame)
        ### Find consecutive buy signals and change classifier so it is filtered out when the classes are balanced. This prevents overfitting.
        consecutive_buys = (omni_tf_data.training_frame['none'] != 1) & (omni_tf_data.training_frame['none'].shift(periods=-1) != 1)
        omni_tf_data.training_frame['none'] += consecutive_buys.astype(int) * 2
        omni_tf_data.time_series_tf_dataset(label_count=omni_params['label_count'], cols_exclude=omni_params['price_count'], window_size=omni_params['hindsight'],
                                   batch_size=omni_params['batch_size'])
    elif omni_params['purpose'] == 'live_prediction':
        omni_tf_data.rename_pred()
        omni_tf_data.pred_frame = omni_tf_data.pred_frame.iloc[:omni_params['hindsight']]
        omni_tf_data.pred_frame = feature_engineering(omni_params, omni_tf_data.pred_frame, realized_predictions=False)
        omni_tf_data.time_series_tf_dataset(label_count=0, window_size=omni_params['hindsight'], batch_size=omni_params['batch_size'])

    return omni_tf_data

def train(training_data = False, omni_params = False):

    if not omni_params:
        from params import omni_params

    if not training_data:
        training_data = data_prep(omni_params)

        with open('models/Optimal Hyperparameters.txt') as hyperparameters:
            omni_params['hyperparams'] = eval(hyperparameters.read())
    else:
        hyperparameters = omni_params['hyperparams']

    omni_params['layers'] = define_layers(omni_params['hyperparams'])

    model, history = train_keras_model(omni_params, training_data)
    model.evaluate(training_data.tf_validation_dataset)

    save_model(omni_params, model, history)

    return model, history

def optimization_evaluation(**parameters):

    input_data = data_prep(glob_params)

    glob_params['hyperparams'] = {}

    for param_name, param_value in parameters.items():
        glob_params['hyperparams'][param_name] = param_value

    _, history = train(training_data = input_data, omni_params = glob_params)

    return history.history['val_accuracy'][-1]

def bayes_optimization():

    from params import omni_params
    global glob_params
    glob_params = omni_params

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
        f=optimization_evaluation,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    optimizer.maximize(init_points=10, n_iter=100)
    print(optimizer.max)

def define_layers(hyperparams):

    layers = [[]] * int(hyperparams['MainLSTMlayers'])
    for layer in range(0, int(hyperparams['MainLSTMlayers'])):
        layers[layer].append([int(hyperparams['MainLSTMNodes']), int(hyperparams['MainLSTMDropout'])])

    return layers

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
        from params import omni_params

    if not os.path.exists('Models/Model Index.csv'):
        print('No Models saved. Train model first.')
        return

    model_index = pd.read_csv('models/Model Index.csv', index_col= 'ID')

    filter = model_index.index > 0
    for specification in omni_params['specs']:
        filter &= omni_params['specs'][specification] == model_index[specification]
    model_ID = model_index.index[filter][0]

    model = load_model('Models/%s.h5' % model_ID)

    predictions = model.predict(input_data.tf_pred_dataset)

    if omni_params['purpose'] == 'live_prediction':
        predictions = pd.DataFrame(predictions, index=[input_data.pred_frame.index.values[0]])
    else:
        predictions = pd.DataFrame(predictions, index=input_data.test_frame.index)
    headers = [a + b for a, b in zip(omni_params['target_tickers'], ([' long'] * len(omni_params['target_tickers'])))]
    headers.append('none')
    predictions.columns = headers

    return predictions

def live_feed():

    from params import omni_params
    omni_params['purpose'] = 'live_prediction'
    interval = p['hindsight_interval'].replace('T','')

    suppress_tf_warnings()
    
    print(f'Printing recommendations every {interval} minutes.')

    while True:

        last_open, EOD, now = dataupdate.last_market_open()
        dataupdate.alphavantage_update(omni_params['tickers'])

        latest_data = data_prep(omni_params)

        prediction = predict(latest_data, omni_params)

        print(Prediction)

        for signal in prediction.values:
            if max(signal) != signal[-1]:
                print('Recommendation to buy ' + omni_params['target_tickers'][0])
            else:
                print('Recommendation not to buy ' + omni_params['target_tickers'][0])


        if last_open != now:
            print("Market Closed")
            break
        time.sleep(60*int(interval))

def train_keras_model(params, input_data):

    model = Sequential()

    hyperparams = params['hyperparams']

    for layer in params['layers'][0]:
        model.add(LSTM(layer[0], activation = 'tanh', return_sequences = True))
        model.add(Dropout(layer[1]))
        model.add(BatchNormalization())


    model.add(LSTM(int(hyperparams['FinalLSTMNodes']), activation ='tanh'))
    model.add(Dropout(hyperparams['FinalLSTMDropout']))
    model.add(BatchNormalization()) 

    model.add(Dense(int(hyperparams['FinalDenseNodes']), activation='relu'))
    model.add(Dropout(hyperparams['FinalDenseDropout']))

    model.add(Dense(params['label_count'], activation=params['activation']))

    opt = tf.keras.optimizers.Adam(lr=hyperparams['LearningRate'], decay=hyperparams['Decay'])

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # NAME = f'{p["target_tickers"][0]}PRED-{p["Hp"]["MainLSTMlayers"]}-MainLayers-{p["Hp"]["MainLSTMNodes"]}-MainNodes-' \
    #     f'{p["Hp"]["FinalLSTMNodes"]}-FinalLSTMNodes-{p["Hp"]["FinalLSTMDropout"]}-FinalLSTMDropout-{p["Hp"]["FinalDenseNodes"]}' \
    #     f'-FinalDenseNodes-{p["Hp"]["FinalDenseDropout"]}-FinalDenseDropout-{p["Hp"]["LearningRate"]}-LearningRate-{p["Hp"]["Decay"]}-Decay'

    # tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


    if params['purpose'] == 'training':
        history = model.fit(input_data.tf_training_dataset, epochs=params['epochs'], validation_data=input_data.tf_validation_dataset) # , callbacks=[tensorboard]
    else:
        history = model.fit(input_data.tf_training_dataset, epochs=params['epochs'])

    return model, history