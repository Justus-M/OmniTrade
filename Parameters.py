class Parameters:
    pass

p = Parameters()

p.TargetTickers = ['AAPL', 'MSFT']
p.epochs = 5
p.Batch_size = 256
p.TestProportion = 0.4
p.ValidationProportion = 0.1
p.hindsight = 32
p.hindsight_interval= '1D'
p.foresight = 8
p.foresight_interval = p.hindsight_interval
p.buy_threshold= 0.02
p.sell_threshold= None
p.y_name = 'close'
p.x_names = ['close', 'volume', 'low', 'high', 'open']
p.tickers = ['AAPL', 'MSFT', 'GOOG', 'IBM']

p.displace = 2
if p.sell_threshold == None:
    p.displace = 1

if p.sell_threshold == None and len(p.TargetTickers) == 1:
    p.activation = 'softmax'
else:
    p.activation = 'sigmoid'

p.LabelCount = len(p.TargetTickers) * p.displace + 1
