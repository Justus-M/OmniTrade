from DataRefresh import DataRefresh
from Parameters import p
from Model import TrainModel
from ApiHelpers import get_daily_stocks
import pandas as pd
import os


tickers = pd.read_csv('Data/SP500.csv')

for ticker in tickers['Symbol']:
    if not os.path.exists('Data/%s-Daily.csv' % (ticker)):
        get_daily_stocks(ticker, '4EHUONPLL0MA0NPU')

DataRefresh(p)
history, TestPredictions, Model, tensor = TrainModel(p)


