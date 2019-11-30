omni_params = {}

##### Input Parameters below
omni_params['target_tickers'] = ['MSFT'] ### tickers of stocks to generate signals for. Matching file must be in the Data/Minute folder
omni_params['tickers'] = ['MSFT'] ### tickers of stocks to use for prediction. ex. ['MSFT', 'AAPL'] Means Microsoft and Apple stock will be used to explain price movements of target_tickers above
omni_params['year_cutoff'] = 2010 ## earliest year from which training data should start. By using 2010 the recession of 2007-2009 is excluded.
omni_params['hindsight'] = 256 ## number of time interval periods behind used to explain price movements. Ex. with 5T (5 minute) interval, we look at the last 256 5-minute intervals to predict the future.
omni_params['hindsight_interval'] = '5T' ## time period interval of data to use - in line with Pandas convention - T for minutes, D for days
omni_params['foresight'] = 32 ## number of time interval periods ahead for prediction. ex. 32 with 5T interval means we look 32*5 = 160 minutes ahead
omni_params['buy_threshold'] = 0.01 ## 0-0.99   minimum price increase for a buy signal. Ex. 0.01 means we generate a buy signal for a price increase of at least 1% after the foresight period
omni_params['sell_threshold'] = None ## None or 0-0.99    minimum price decrease for a short signal
omni_params['data_path'] = 'Data/Minute' ## Path where your market data is stored. Should be stored as 'ticker.csv' ex. 'MSFT.csv'
omni_params['validation_proportion'] = 0.2 ## Proportion of data to use for model validation
omni_params['epochs'] = 3 ## number of neural network training epochs
omni_params['batch_size'] = 128 ## batch size for neural network training

#Below is the search space for the automated hyperparameter optimization
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

omni_params['hindsight_extension'] = None #[1, 2, 3, 4, 8]
omni_params['purpose'] = 'training'
omni_params['test_proportion'] = 0.2

for ticker in omni_params['target_tickers']:
    if ticker not in omni_params['tickers']:
        omni_params['tickers'].append(ticker)

omni_params['target_tickers'].sort()
omni_params['tickers'].sort()
omni_params['label_count'] = len(omni_params['target_tickers']) * omni_params['displace'] + 1
omni_params['price_count'] = len(omni_params['target_tickers']) * 2
specs = {}
specs['target_tickers'] = str(omni_params['target_tickers'])
specs['tickers'] = omni_params['tickers']
specs['n_tickers'] = len(omni_params['tickers'])
specs['foresight'] = omni_params['foresight']
specs['hindsight'] = omni_params['hindsight']
specs['interval'] = omni_params['hindsight_interval']
specs['short'] = omni_params['sell_threshold']
omni_params['specs'] = specs
