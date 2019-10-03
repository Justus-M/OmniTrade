from Helpers import BuildTickerList

p = dict()

p['headers'] = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
p = BuildTickerList(p)
p['TargetTickers'] = ['MSFT']
p['tickers'].sort()
p['TargetTickers'].sort()
p['epochs'] = 5
p['Batch_size'] = 256
p['TestProportion'] = 0.01
p['ValidationProportion'] = 0.3
p['hindsight'] = 256
p['HindsightExtension'] = None #[1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
p['hindsight_interval'] = '5T'
p['foresight'] = 16
p['buy_threshold'] = 0.005
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

