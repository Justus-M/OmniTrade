import csv
import pandas as pd
import os
from datetime import datetime, date, timedelta
import ApiHelpers
import requests
from bs4 import BeautifulSoup

# https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

AlphaVantageKey = '4EHUONPLL0MA0NPU'
Lagdays = 7
SP500 = ApiHelpers.get_SP500_list()

for ticker in SP500:
    if os.path.exists('Data/%s-Daily.csv' % (ticker)):
        with open('Data/%s-Daily.csv' % (ticker)) as f:
            reader = csv.DictReader(f)
            row1 = next(reader)
            date = datetime.strptime(row1['timestamp'], '%Y-%m-%d')

            if date < datetime.today() - timedelta(days=Lagdays) and date > datetime.today() - timedelta(days=Lagdays + 60):
                ApiHelpers.get_daily_stocks(ticker, AlphaVantageKey)
    else:
        ApiHelpers.get_daily_stocks(ticker, '4EHUONPLL0MA0NPU')
        temp = pd.read_csv('Data/%s-Daily.csv' % (ticker))
    temp = pd.read_csv('Data/%s-Daily.csv' % (ticker))
    if len(temp) < 5:
        os.remove('Data/%s-Daily.csv' % (ticker))
        print(ticker + ' error')









