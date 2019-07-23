import requests
import csv
import pandas as pd
import time


def get_trades(ticker, ticker2, start=1420066800000, limit = 5000):
    csv_path = r"data/trades/%s-%s.csv" % (ticker, ticker2)
    end = 1546297200000 # 01/01/2019
    last_timestamp = get_last_timestamp(csv_path) if get_last_timestamp(csv_path) > start else start
    while last_timestamp < end:
        time.sleep(6)
        r = requests.get('https://api-pub.bitfinex.com/v2/trades/t%s%s/hist?limit=%d&start=%d&sort=1'
                         % (ticker, ticker2, limit, last_timestamp))
        lines = r.json()
        print(last_timestamp)
        with open(csv_path, 'a') as myfile:
            wr = csv.writer(myfile)
            for l in lines:
                wr.writerow(l)
                last_timestamp = int(l[1])


def get_last_timestamp(file_path):
    df = pd.read_csv(file_path, header=0, names=["ID", "MTS", "AMOUNT", "PRICE"])
    return df['MTS'].max()


get_trades('BTC', 'USD', 1420066800000)
