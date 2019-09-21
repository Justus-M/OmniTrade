import pandas as pd
import os
from datetime import datetime, timedelta
import ApiHelpers

p = dict()
tickers = ApiHelpers.get_SP500_list()
p['tickers'] = []
for ticker in tickers:
     if os.path.exists('Data/%s-Daily.csv' % (ticker)):
         temp = pd.read_csv('Data/%s-Daily.csv' % (ticker), header=0,
                            names=['count', 'time', 'open', 'high', 'low', 'close', 'volume'])
         temp = pd.to_datetime(temp['time'])
         if temp.iloc[-1]<pd.to_datetime(datetime(2009, 1, 1)) and temp.iloc[1]>datetime.today()-timedelta(days=14):
             p['tickers'].append(ticker)

p['TargetTickers'] = ['MSFT']
p['TargetTickers'].sort()
p['headers'] = ['count', 'time', 'open', 'high', 'low', 'close', 'volume']
p['epochs'] = 5
p['Batch_size'] = 256
p['TestProportion'] = 0.3
p['ValidationProportion'] = 0.3
p['hindsight'] = 64
p['hindsight_interval'] = '1D'
p['foresight'] = 16
p['foresight_interval'] = p['hindsight_interval']
p['buy_threshold'] = 0.03
p['sell_threshold'] = None
p['y_name'] = 'close'
p['x_names'] = ['close', 'volume', 'low', 'high', 'open']
p['layers'] = [[128, 0.3], [128, 0.3]]
p['Purpose'] = 'Training'

p['displace'] = 2
if p['sell_threshold'] == None:
    p['displace'] = 1

if p['sell_threshold'] == None and len(p['TargetTickers']) == 1:
    p['activation'] = 'softmax'
else:
    p['activation'] = 'sigmoid'

p['LabelCount'] = len(p['TargetTickers']) * p['displace'] + 1
