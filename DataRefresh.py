import pandas as pd

def DataRefresh(p):
    frames = dict.fromkeys(p.tickers)

    baseindex = pd.read_csv('Data/' + p.TargetTickers[0] + '-Daily.csv',header = 0, names=['count', 'time', 'open', 'high', 'low', 'close', 'volume'],index_col = 'time', parse_dates=True)
    baseindex.index = pd.to_datetime(baseindex.index, unit='D')
    baseindex = baseindex.resample(p.hindsight_interval).mean()

    All = pd.DataFrame(index = baseindex.index)
    del baseindex

    for ticker in p.tickers:
        frame = pd.read_csv('Data/' + ticker + '-Daily.csv',header = 0, names=['count', 'time', 'open', 'close', 'high', 'low', 'volume'],index_col = 'time', parse_dates=True)
        frame = frame[p.x_names] #only keep relevant columns
        frame.index = pd.to_datetime(frame.index, unit = 'D')
        frame.columns = ticker + ' ' + frame.columns.values
        frame = frame.resample(p.hindsight_interval).mean()
        All = All.merge(frame, how='inner', left_index=True, right_index=True)

    All = All.loc[All.index.drop_duplicates()]
    All.dropna(inplace=True)

    # for ticker in p.TargetTickers:
    #     trades = pd.read_csv('Data/' + ticker + '-USD-trades.csv', names=['ID', 'time', ticker + ' AMOUNT', 'PRICE'],
    #                          index_col='time', parse_dates=True)
    #     trades.index = pd.to_datetime(trades.index, unit='ms')
    #     trades = trades.resample(p.hindsight_interval).sum()
    #     All = All.merge(trades[ticker + ' AMOUNT'], how='inner', left_index=True, right_index=True)
    #     All.dropna(inplace=True)
    # del trades

    All.to_csv('Data/Processed.csv')

