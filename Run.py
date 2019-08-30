from DataRefresh import DataRefresh
from Parameters import p
from Model import TrainModel
from ApiHelpers import get_daily_stocks
import pandas as pd
import os


tickers = pd.read_csv('SP500.csv')

for ticker in tickers['Symbol']:
    if not os.path.exists('Data/%s-Daily.csv' % (ticker)):
        get_daily_stocks(ticker, '4EHUONPLL0MA0NPU')
        temp = pd.read_csv('Data/%s-Daily.csv' % (ticker))
        if len(temp) < 5:
            tickers = tickers[tickers['Symbol'] != ticker]
            tickers.to_csv('SP500.csv')
            os.remove('Data/%s-Daily.csv' % (ticker))


DataRefresh(p)
history, Model, tensor, TestPredictions, test = TrainModel(p)

testing = Model.predict(tensor)
print(len(testing))


