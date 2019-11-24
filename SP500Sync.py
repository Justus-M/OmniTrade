import pandas as pd
import os
from datetime import datetime, timedelta, time
import time as t
import ApiHelpers
import pytz
from Helpers import CsvEndReader
import requests



def Main():

    AlphaVantageKey = '4EHUONPLL0MA0NPU'

    tz = pytz.timezone('America/New_York')

    LastOpen, EOD, now = GetLastOpen()

    log = {'NoOverlap':[], 'timestamp':[]}

    uptodate = 0 #count tickers which are already up to date
    requestlimiter = 0 #used to track time since last request to limit requests per minute

    tickers = []
    for ticker in os.listdir('Data/Iso'): #build list of tickers
        if 'csv' in ticker:
            tickers.append(ticker.replace('.csv', ''))

    for ticker in tickers:
        file = 'Data/Iso/%s.csv' % (ticker)
        timestamp = CsvEndReader(file, 1, Processed = True).index.values[0]
        pdtimestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S')
        timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz)

        if timestamp >= EOD or (timestamp >= LastOpen): #check if it's up to date
            uptodate += 1
            continue

        else:
            t.sleep(max(0, 11 - (t.time() - requestlimiter))) #limit API requests to every 10 seconds to prevent block
            # minutes, stamps = ApiHelpers.get_minute_stocks(ticker, AlphaVantageKey, outputsize='full')
            update = ApiHelpers.get_minute_data(ticker, AlphaVantageKey)
            requestlimiter = t.time()

        if pdtimestamp in update.index[1:]: #make sure there is overlap and no gap between the update and the current data
            update = update.loc[:pdtimestamp]
            update = update.iloc[::-1]
            update.to_csv(file, header = False, mode='a')

        else:
            print('No overlap for ' + ticker)
            log['NoOverlap'].append(ticker)
            log['timestamp'].append(str(datetime.now()))

    print(str(uptodate) + ' tickers already up to date.')
    log = pd.DataFrame({k: pd.Series(l) for k, l in log.items()})
    log.to_csv('SyncLog.csv') #save log of unsuccesful updates for manual intervention/investigation

def iex():
    tickers = []
    for ticker in os.listdir('Data/Iso'):  # build list of tickers
        if 'csv' in ticker:
            tickers.append(ticker.replace('.csv', ''))
    for ticker in os.listdir('Data/iex'):  # build list of tickers
        if 'csv' in ticker:
            if ticker.replace('.csv', '') in tickers:
                tickers.remove(ticker.replace('.csv', ''))

    for ticker in tickers:
        try:
            re = requests.get('https://cloud.iexapis.com/stable/stock/%s/chart/5dm?token=sk_f9b67e6efa7c42fea5d1d03db48d92cc' % ticker)
            frame = pd.read_json(re.content)
            frame['timestamp'] = frame['date'].apply(str).replace('00:00:00', '', regex=True) + frame['minute'] + ':00'
            frame.set_index('timestamp', inplace=True)
            frame = frame[['open', 'high', 'low', 'close', 'volume', 'numberOfTrades']]
            frame.to_csv('Data/iex/%s.csv' % ticker)
        except:
            print(ticker + 'failed')

def GetLastOpen():

    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    EOD = datetime(now.year, now.month, now.day, 16, 00).replace(tzinfo=tz)

    if now.time() > time(16, 00, 00) or now.time() < time(9, 30, 00) or now.weekday() > 4:
        MarketClosed = True
    else:
        MarketClosed = False

    if MarketClosed:
        if now.weekday() < 5 and now.time() > time(16, 00, 00):
            LastOpen = EOD
        else:
            if now.weekday() == 0:
                prev = 3
            elif now.weekday() == 6:
                prev = 2
            else:
                prev = 1
            LastOpen = EOD - timedelta(days=prev)
    else:
        LastOpen = now
    return LastOpen, EOD, now

if __name__ == '__main__':

    while True:
        LastOpen, EOD, now = GetLastOpen()
        # iex()

        while LastOpen == now:
            t.sleep(3600)
            LastOpen, EOD, now = GetLastOpen()

        Main()
        LastOpen, EOD, now = GetLastOpen()

        while LastOpen != now:
            t.sleep(3600)
            LastOpen, EOD, now = GetLastOpen()