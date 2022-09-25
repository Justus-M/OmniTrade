import pandas as pd
import os

tickers = []

for ticker in os.listdir('Data/Filler'):  # build list of tickers
    if 'csv' in ticker:
        tickers.append(ticker.replace('.csv', ''))

for ticker in tickers:
    filler = pd.read_csv(f'{ticker}.csv')
    filler['timestamp'] = pd.to_datetime(filler["<DATE>"].astype(str) + ' ' +  filler['<TIME>'], format='%Y-%m-%d %H:%M:%S')
    filler.set_index('timestamp', inplace = True)
    filler = filler[filler.columns.values[2:]]
    filler.columns = ['open', 'high', 'low', 'close', 'volume']
    current = pd.read_csv(f'Data/Minute/{ticker}.csv', header=0, index_col ='timestamp')
    current.index = pd.to_datetime(current.index, unit='s')
    last = current.index[-1]
    current = current[:-1].append(filler.loc[last:])
    current.to_csv(f'Data/Minute/{ticker}.csv')



