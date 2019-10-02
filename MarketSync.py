import csv
import pandas as pd
import os
from datetime import datetime, date, timedelta
import ApiHelpers
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import random
import DataPrep

keys = ['SL07U4SE8UDRPBT4', 'N1Q616CI18BURJUS', '82Q2JYA7VUSB379H', 'JQ6TTTARU85VYTC5', '4WRWRSCJTRAJPJLU', 'X404EWZN9T26RRQU']
keys = iter(keys)
AlphaVantageKey = next(keys)
Lagdays = 7
SP500 = ApiHelpers.get_SP500_list()
frequency = 'Daily'

for ticker in SP500:
    ticker = ticker.replace('.', '')
    if os.path.exists('Data/Day/%s.csv' % (ticker)):
        with open('Data/Day/%s.csv' % (ticker)) as f:
            reader = csv.DictReader(f)
            row1 = next(reader)
            date = datetime.strptime(row1['timestamp'], '%Y-%m-%d')
            if date < datetime.today() - timedelta(days=Lagdays) and date > datetime.today() - timedelta(days=Lagdays + 60):
                ApiHelpers.get_daily_stocks(ticker, AlphaVantageKey)
    else:
        ApiHelpers.get_daily_stocks(ticker, AlphaVantageKey)
        temp = pd.read_csv('Data/Day/%s.csv' % (ticker))
    temp = pd.read_csv('Data/Day/%s.csv' % (ticker))
    if len(temp) < 5:
        os.remove('Data/Day/%s.csv' % (ticker))
        print(ticker + ' error')


p['hindsight_interval'] = '1D'
p['foresight_interval'] = p['hindsight_interval']
DataPrep.DataPreparation(p)









