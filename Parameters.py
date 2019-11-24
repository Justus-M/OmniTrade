p = {}

##### Input Parameters below
p['TargetTickers'] = ['MSFT']
p['Tickers'] = ['MSFT', 'CRM', 'ORC']
p['YearCutoff'] = 2012 ## earliest year from which training data starts
p['hindsight'] = 256
p['HindsightExtension'] = None #[1, 2, 3, 4, 8]
p['hindsight_interval'] = '5T'
p['foresight'] = 32 ## number of time interval periods ahead for prediction
p['buy_threshold'] = 0.01 ## minimum return for a buy signal
p['sell_threshold'] = None ## minimum return for a sell signal
p['Purpose'] = 'Training'

p['TestProportion'] = 0.2
p['ValidationProportion'] = 0.2
p['epochs'] = 3
p['Batch_size'] = 128
p['activation'] = 'sigmoid'
### Input Parameters above


### Do not modify below
p['displace'] = 2
if p['sell_threshold'] == None:
    p['displace'] = 1

if p['sell_threshold'] == None and len(p['TargetTickers']) == 1:
    p['activation'] = 'softmax'
else:
    p['activation'] = 'sigmoid'


for ticker in p['TargetTickers']:
    if ticker not in p['Tickers']:
        p['Tickers'].append(ticker)

p['TargetTickers'].sort()
p['Tickers'].sort()
p['LabelCount'] = len(p['TargetTickers']) * p['displace'] + 1
