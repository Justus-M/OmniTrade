import pandas as pd
import os
from datetime import datetime, timedelta, time
import time as t
from apihelpers import get_minute_data
import pytz
from Helpers import csv_end_reader
from keys import my_alpha_vantage_key

tz = pytz.timezone('America/New_York')

def alphavantage_update(tickers = []):

    alpha_vantage_key = my_alpha_vantage_key

    last_open, EOD, now = last_market_open()

    log = {'no_overlap':[], 'timestamp':[]}

    up_to_date = 0 #count tickers which are already up to date
    request_limiter = 0 #used to track time since last request to limit requests per minute


    if not tickers:
        for ticker in os.listdir('Data/Minute'): #build list of tickers
            if 'csv' in ticker:
                tickers.append(ticker.replace('.csv', ''))

    for ticker in tickers:
        file = f'Data/Minute/{ticker}.csv'
        timestamp = csv_end_reader(file, 1, Processed = True).index.values[0]
        pandas_timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S')
        timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz)

        if timestamp >= EOD or (timestamp >= last_open): #check if it's up to date
            up_to_date += 1
            continue

        else:
            t.sleep(max(0, 11 - (t.time() - request_limiter))) #limit API requests to every 10 seconds to prevent block
            # minutes, stamps = ApiHelpers.get_minute_stocks(ticker, alpha_vantage_key, outputsize='full')
            update = get_minute_data(ticker, alpha_vantage_key)
            request_limiter = t.time()

        if pandas_timestamp in update.index[1:]: #make sure there is overlap and no gap between the update and the current data
            update = update.loc[:pandas_timestamp]
            update = update.iloc[::-1]
            update.to_csv(file, header = False, mode='a')

        else:
            print(f'No overlap for {ticker}, not updating data for this ticker to avoid data gap.')
            log['no_overlap'].append(ticker)
            log['timestamp'].append(str(datetime.now()))

    log = pd.DataFrame({k: pd.Series(l) for k, l in log.items()})
    log.to_csv('sync_log.csv') #save log of unsuccesful updates for manual intervention/investigation

def last_market_open():

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
        LastOpen, EOD, now = last_market_open()

        while LastOpen == now:
            t.sleep(3600)
            LastOpen, EOD, now = last_market_open()

        alphavantage_update()

        LastOpen, EOD, now = last_market_open()

        while LastOpen != now:
            t.sleep(3600)
            LastOpen, EOD, now = last_market_open()