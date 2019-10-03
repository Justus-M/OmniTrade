import pandas as pd
import time
import os
import csv
from Helpers import CsvEndReader

def DataPreparation(p):

    start = time.time()
    intermittent = time.time()

    baseindex = pd.read_csv('Data/Minute/AAPL.csv', header=0, index_col = 'timestamp', parse_dates=True)
    baseindex.index = pd.to_datetime(baseindex.index, unit='s')
    baseindex = baseindex.resample(p['hindsight_interval']).mean()

    Batch = pd.DataFrame(index = baseindex.index)

    count = 0
    batches = []

    for ticker in p['tickers']:
        frame = pd.read_csv('Data/Minute/' + p['TargetTickers'][0] + '.csv', header = 0, index_col = 'timestamp', parse_dates=True)
        frame = frame[['open', 'high', 'low', 'close', 'volume']] #only keep relevant columns
        frame.index = pd.to_datetime(frame.index, unit = 's')
        frame.columns = ticker + ' ' + frame.columns.values
        frame = frame.resample(p['hindsight_interval']).mean()
        Batch = Batch.merge(frame, how='inner', left_index=True, right_index=True)

        count+=1

        if count%20 == 0:
            batches.append(count)
            Batch.to_csv(str(count) + '.csv')
            Batch = pd.DataFrame(index=baseindex.index)

        print(ticker + ' ' + str(time.time() - intermittent))
        intermittent = time.time()

    batches.append(count)
    Batch.to_csv(str(count) + '.csv')
    del Batch

    All = pd.DataFrame(index=baseindex.index)

    for batch in batches:
        Batch = pd.read_csv(str(batch) + '.csv', header = 0, index_col = 'timestamp', parse_dates=True)
        All = All.merge(Batch, how='inner', left_index=True, right_index=True)
        os.remove(str(batch) + ".csv")

    del Batch
    del baseindex

    All = All.loc[All.index.drop_duplicates()]
    All.dropna(inplace=True)
    date = str(All.index.values[0])

    All.to_csv('Data/Processed%s.csv' % p['hindsight_interval'], index_label = 'timestamp', chunksize = 100000)

    print('Processing done after' + ' ' + str(time.time() - start) + ' seconds')

def DataUpdate(p):

    start = time.time()
    if os.path.exists('Data/Processed%s.csv' % p['hindsight_interval']):
        lastminute = CsvEndReader('Data/Processed%s.csv' % p['hindsight_interval'], 1, Processed = True).index.values[0]

        update = pd.DataFrame()
        for ticker in p['tickers']:
            tickerupdate = CsvEndReader('Data/Minute/%s.csv' % ticker, 1500)
            try:
                cutoff = tickerupdate.index.get_loc(lastminute)
            except:
                print('update error ' + ticker)
                return
            tickerupdate.index = pd.to_datetime(tickerupdate.index)
            tickerupdate = tickerupdate.iloc[cutoff+1:]
            tickerupdate.columns = ticker + ' ' + tickerupdate.columns.values
            tickerupdate = tickerupdate.astype(float)
            tickerupdate = tickerupdate.resample(p['hindsight_interval']).mean()
            if len(tickerupdate)<100:
                print(ticker)
            if ticker == p['tickers'][0]:
                update = tickerupdate
            else:
                update = update.merge(tickerupdate, how='inner', left_index=True, right_index=True)

        update.to_csv('Data/Processed%s.csv' % p['hindsight_interval'], header = False, mode = 'a')
        print(update)

        print('Update done after' + ' ' + str(time.time() - start) + ' seconds')

    else:
        print('Processed data does not exist for frequency ' + p['hindsight_interval'])






