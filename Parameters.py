import pandas as pd
import os
from datetime import datetime, timedelta
import ApiHelpers
import csv
from CsvEndReader import CsvEndReader

p = dict()

p['headers'] = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
p = BuildTickerList(p)
p['TargetTickers'] = ['AAPL', 'MSFT', 'CRM']
p['tickers'].sort()
p['TargetTickers'].sort()
p['epochs'] = 5
p['Batch_size'] = 256
p['TestProportion'] = 0.3
p['ValidationProportion'] = 0.3
p['hindsight'] = 512
p['HindsightExtension'] = None #[1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
p['hindsight_interval'] = '5T'
p['foresight'] = 32
p['foresight_interval'] = p['hindsight_interval']
p['buy_threshold'] = 0.01
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

def BuildTickerList(p):

    tickers = []
    Exclude = pd.read_csv('Synclog.csv')['NoOverlap']

    for ticker in os.listdir('Data/Minute'):
        if 'csv' in ticker and ticker.replace('.csv', '') not in Exclude:
            tickers.append(ticker.replace('.csv', ''))

    startdate = []
    for ticker in tickers:
        row1 = pd.read_csv('Data/Minute/%s.csv' % ticker, nrows=1)
        startdate.append([ticker, row1['timestamp'][0]])

    startdate = pd.DataFrame(startdate, columns=['ticker', 'startdate']).set_index('ticker')

    p['tickers'] = []
    lateststart = pd.to_datetime(datetime(2009, 1, 1))
    earliestend = datetime.today() - timedelta(days=7)
    for ticker in tickers:

        tickerstart = pd.to_datetime(startdate['startdate'].loc[ticker])
        tickerend = pd.to_datetime(CsvEndReader('Data/Minute/%s.csv' % (ticker), 1).index.values[0])

        if tickerstart < lateststart and tickerend > earliestend:
            p['tickers'].append(ticker)
    return p