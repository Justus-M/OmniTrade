import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

def get_daily_stocks(ticker, key):
    print(ticker)

    temp = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s&datatype=csv'
        % (ticker, key))

    time.sleep(10)
    temp.to_csv('Data/Day/%s.csv' % (ticker))

def get_daily_adjusted_stocks(ticker, outputsize = 'full'):
    print(ticker)
    temp = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=%s&outputsize=%s&apikey=%s&datatype=csv'
                     % (ticker, outputsize, key))
    time.sleep(10)
    temp.to_csv('Data/Adjusted/%s-Daily-Adjusted.csv' % (ticker))

def get_SP500_list():
    sp = requests.get('https://www.slickcharts.com/sp500')
    sp = BeautifulSoup(sp.content, 'html.parser')

    SP500 = []

    for each in sp.find_all('a', href=True):
        if '/symbol/' in each.get('href'):
            SP500.append(each.get('href').split('/symbol/')[1].upper())

    SP500 = list(dict.fromkeys(SP500))

    return SP500

def get_minute_data(ticker, key):


    try:
        frame = pd.read_csv(
            'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=%s&outputsize=full&apikey=%s&datatype=csv' % (
            ticker, '1min', key))
        frame.set_index('timestamp', inplace = True)
        frame.index = pd.to_datetime(frame.index.values, format='%Y-%m-%d %H:%M:%S')
    except:
        print('Request error for ' + ticker)


    return frame











