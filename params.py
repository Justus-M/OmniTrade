omni_params = {}

##### Input Parameters below
omni_params['target_tickers'] = ['MSFT']
omni_params['tickers'] = ['MSFT']
omni_params['year_cutoff'] = 2010 ## earliest year from which training data starts
omni_params['hindsight'] = 256
omni_params['hindsight_extension'] = None #[1, 2, 3, 4, 8]
omni_params['hindsight_interval'] = '5T'
omni_params['foresight'] = 32 ## number of time interval periods ahead for prediction
omni_params['buy_threshold'] = 0.01 ## minimum return for a buy signal
omni_params['sell_threshold'] = None ## minimum return for a sell signal
omni_params['purpose'] = 'training'
omni_params['data_path'] = 'Data/Minute'
omni_params['test_proportion'] = 0.2
omni_params['validation_proportion'] = 0.2
omni_params['epochs'] = 3
omni_params['batch_size'] = 128
omni_params['activation'] = 'sigmoid'

omni_params['bayesian_search_space'] = {'MainLSTMlayers': (2, 8),
           'MainLSTMNodes': (32, 256),
           'MainLSTMDropout': (0.15, 0.4),
           'FinalLSTMNodes': (32, 256),
           'FinalLSTMDropout': (0.15, 0.4),
           'FinalDenseNodes': (32, 256),
           'FinalDenseDropout': (0.15, 0.4),
           'LearningRate': (0.0001, 0.01),
           'Decay': (0.00000001, 0.01)
          }
omni_params['bayesian_initial_points'] = 5
omni_params['bayesian_iterations'] = 20
### Input Parameters above


### Do not modify below
omni_params['displace'] = 2
if omni_params['sell_threshold'] == None:
    omni_params['displace'] = 1

if omni_params['sell_threshold'] == None and len(omni_params['target_tickers']) == 1:
    omni_params['activation'] = 'softmax'
else:
    omni_params['activation'] = 'sigmoid'


for ticker in omni_params['target_tickers']:
    if ticker not in omni_params['tickers']:
        omni_params['tickers'].append(ticker)

omni_params['target_tickers'].sort()
omni_params['tickers'].sort()
omni_params['label_count'] = len(omni_params['target_tickers']) * omni_params['displace'] + 1
omni_params['price_count'] = (omni_params['label_count'] - 1) * 2
specs = {}
specs['target_tickers'] = str(omni_params['target_tickers'])
specs['tickers'] = omni_params['tickers']
specs['n_tickers'] = len(omni_params['tickers'])
specs['foresight'] = omni_params['foresight']
specs['hindsight'] = omni_params['hindsight']
specs['interval'] = omni_params['hindsight_interval']
omni_params['specs'] = specs
