import csv
import pandas as pd
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from fake_useragent import UserAgent
import random
import io

def get_trades(ticker, ticker2, start=1522015200000, limit = 5000):
    csv_path = r"data/trades/%s-%s.csv" % (ticker, ticker2)
    end = 1564005600000 # 01/01/2019
    last_timestamp = get_last_timestamp(csv_path) if get_last_timestamp(csv_path) > start else start
    while last_timestamp < end:
        time.sleep(2)
        r = requests.get('https://api-pub.bitfinex.com/v2/trades/t%s%s/hist?limit=%d&start=%d&sort=1'
                         % (ticker, ticker2, limit, last_timestamp))
        lines = r.json()
        print(str(last_timestamp) + " " + str(datetime.fromtimestamp(last_timestamp/1000.0)) + " for " + ticker)
        with open(csv_path, 'a') as myfile:
            wr = csv.writer(myfile)
            for l in lines:
                wr.writerow(l)
                last_timestamp = int(l[1])


def get_last_timestamp(file_path):
    df = pd.read_csv(file_path, header=0, names=["time", "open", "close", "high", "low", "volume"])
    return df['time'].max()

def get_Candles(ticker, ticker2, start=1546297200000, limit = 5000):
    csv_path = r"Data/%s-%s-train.csv" % (ticker, ticker2)
    end = 1564005600000 # 01/01/2019
    last_timestamp = get_last_timestamp(csv_path) if get_last_timestamp(csv_path) > start else start
    print(str(last_timestamp) + " " + str(datetime.fromtimestamp(last_timestamp/1000.0)) + " for " + ticker)
    while last_timestamp < end:
        time.sleep(0.8)
        r = requests.get('https://api-pub.bitfinex.com/v2/candles/trade:1m:t%s%s/hist?limit=%d&start=%d&sort=1'
                         % (ticker, ticker2, limit, last_timestamp))
        lines = r.json()
        print(str(last_timestamp) + ticker)
        with open(csv_path, 'a') as myfile:
            wr = csv.writer(myfile)
            for l in lines:
                wr.writerow(l)
                last_timestamp = int(l[0])


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

def Intrinio():
    key = 'OmM5NzQ5NDAzZjU4MzVmNmY5OWI1Zjc1MTdlNWZlMjk3'

def get_minute_stocks(ticker, key, outputsize = 'full'):
    stamps = []
    minutes = 0
    try:
        minutes = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=1min&outputsize=%s&apikey=%s'
                     % (ticker, outputsize, key)).json()
    except:
        print(ticker + ' error')
        return minutes, stamps
    try:
        minutes['Time Series (1min)']
    except:
        print(ticker + ' error')
        return minutes, stamps

    for stamp in minutes['Time Series (1min)']:
        stamps.append(stamp)

    return minutes['Time Series (1min)'], stamps








