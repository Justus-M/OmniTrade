import pandas as pd
import os
from datetime import datetime, timedelta, time
import time as t
import ApiHelpers
import pytz
from Helpers import CsvEndReader



def Main():

    AlphaVantageKey = '4EHUONPLL0MA0NPU'
    mapping = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'}

    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    EOD = datetime(now.year, now.month, now.day, 16, 00).replace(tzinfo=tz)
    LastOpen = GetLastOpen(EOD, now)

    log = {'NoOverlap':[], 'timestamp':[], 'Missing':[]}

    uptodate = 0 #count tickers which are already up to date
    requestlimiter = 0 #track time since last request to limit requests per minute

    tickers = []
    for ticker in os.listdir('Data/Minute'): #build list of tickers
        if 'csv' in ticker:
            tickers.append(ticker.replace('.csv', ''))

    for ticker in tickers:
        file = 'Data/Minute/%s.csv' % (ticker)
        timestamp = CsvEndReader(file, 1, Processed = True).index.values[0]
        timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz)

        if timestamp >= EOD or (timestamp >= LastOpen): #check if it's up to date
            uptodate += 1
            continue

        else:
            t.sleep(max(0, 10 - (t.time() - requestlimiter))) #limit API requests to every 10 seconds to prevent block
            minutes, stamps = ApiHelpers.get_minute_stocks(ticker, AlphaVantageKey, outputsize='full')
            requestlimiter = t.time()

        if timestamp.strftime('%Y-%m-%d %H:%M:%S') in stamps: #make sure there is overlap and no gap between the update and the current data
            data = CsvEndReader(file, 1).astype(float)
            cutoff = stamps.index(data.index.values[0])

            for stamp in stamps[0:cutoff]:
                updatefeed = pd.DataFrame(index=[stamp], columns=data.columns.values)

                for col in mapping:
                    updatefeed[mapping[col]].iloc[0] = float(minutes[stamp][col]) #unpack JSON from api request into dataframe

                updatefeed.index.name = 'timestamp'
                data = data.append(updatefeed)

            data = data.iloc[1:][::-1] #remove line from database and reverse the order or the frame to match the current data order
            data.to_csv(file, header = False, mode='a')

        else:
            print('No overlap for ' + ticker)
            log['NoOverlap'].append(ticker)
            log['timestamp'].append(str(datetime.now()))

    print(str(uptodate) + ' tickers already up to date.')
    log = pd.DataFrame({k: pd.Series(l) for k, l in log.items()})
    log.to_csv('SyncLog.csv') #save log of unsuccesful updates for manual intervention/investigation

def GetLastOpen(EOD, now):

    if now.time() > time(16, 00, 00) or now.time() < time(9, 00, 00) or now.weekday() > 4:
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
    return LastOpen

if __name__ == '__main__':

    Main()