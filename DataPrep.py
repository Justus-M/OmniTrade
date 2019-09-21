import pandas as pd

def DataPreparation(p):
    frames = dict.fromkeys(p['tickers'])

    baseindex = pd.read_csv('Data/' + p['TargetTickers'][0] + '-Daily.csv',header = 0, names=p['headers'],index_col = 'time', parse_dates=True)
    baseindex.index = pd.to_datetime(baseindex.index, unit='D')
    baseindex = baseindex.resample(p['hindsight_interval']).mean()

    All = pd.DataFrame(index = baseindex.index)
    del baseindex

    for ticker in p['tickers']:
        frame = pd.read_csv('Data/' + ticker + '-Daily.csv',header = 0, names=p['headers'],index_col = 'time', parse_dates=True)
        frame = frame[p['x_names']] #only keep relevant columns
        frame.index = pd.to_datetime(frame.index, unit = 'D')
        frame.columns = ticker + ' ' + frame.columns.values
        frame = frame.resample(p['hindsight_interval']).mean()
        All = All.merge(frame, how='inner', left_index=True, right_index=True)

    All = All.loc[All.index.drop_duplicates()]
    All.dropna(inplace=True)
    All = All.iloc[::-1]

    All.to_csv('Data/Processed%s.csv' % (p['hindsight_interval']))

