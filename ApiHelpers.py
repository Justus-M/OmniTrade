import requests
import csv
import pandas as pd
import time
from datetime import datetime


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
    df = pd.read_csv(file_path, header=0, names=["ID", "MTS", "AMOUNT", "PRICE"])
    return df['MTS'].max()

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

get_trades('BTC', 'USD', 1420066800000)
get_trades('ETH', 'USD', 1420066800000)
get_trades('LTC', 'USD', 1420066800000)
get_trades('EOS', 'USD', 1420066800000)