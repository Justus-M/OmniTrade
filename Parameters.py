import pandas as pd
import os
from datetime import datetime

p = dict()

tickers = pd.read_csv('SP500.csv')
p['tickers'] = []

for ticker in tickers['Symbol']:
     if os.path.exists('Data/%s-Daily.csv' % (ticker)):
         temp = pd.read_csv('Data/%s-Daily.csv' % (ticker), header=0,
                            names=['count', 'time', 'open', 'high', 'low', 'close', 'volume'])
         temp = pd.to_datetime(temp['time'])
         if temp.iloc[-1]<pd.to_datetime(datetime(2009, 1, 1)) and temp.iloc[1]>pd.to_datetime(datetime(2019, 8, 15)):
             p['tickers'].append(ticker)

p['TargetTickers'] = ['MSFT', 'AAPL', 'CRM', 'IBM']
p['epochs'] = 5
p['Batch_size'] = 256
p['TestProportion'] = 0.2
p['ValidationProportion'] = 0.1
p['hindsight'] = 64
p['hindsight_interval'] = '1D'
p['foresight'] = 16
p['foresight_interval'] = p['hindsight_interval']
p['buy_threshold'] = 0.03
p['sell_threshold'] = None
p['y_name'] = 'close'
p['x_names'] = ['close', 'volume', 'low', 'high', 'open']

p['displace'] = 2
if p['sell_threshold'] == None:
    p['displace'] = 1

if p['sell_threshold'] == None and len(p['TargetTickers']) == 1:
    p['activation'] = 'softmax'
else:
    p['activation'] = 'sigmoid'

p['LabelCount'] = len(p['TargetTickers']) * p['displace'] + 1
