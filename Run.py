from DataRefresh import DataRefresh
from Parameters import p
from Model import TrainModel
from ApiHelpers import get_daily_stocks
import pandas as pd
import os
import TradeSimulation
import DataPrep


# tickers = pd.read_csv('SP500.csv')
#
# for ticker in tickers['Symbol']:
#     if not os.path.exists('Data/%s-Daily.csv' % (ticker)):
#         get_daily_stocks(ticker, '4EHUONPLL0MA0NPU')
#         temp = pd.read_csv('Data/%s-Daily.csv' % (ticker))
#         if len(temp) < 5:
#             tickers = tickers[tickers['Symbol'] != ticker]
#             tickers.to_csv('SP500.csv')
#             os.remove('Data/%s-Daily.csv' % (ticker))

DataRefresh(p)

DFrame = pd.read_csv('Data/Processed.csv', index_col = 'time', parse_dates=True)

testlength = int(p['TestProportion']*len(DFrame))
TestFrame = DFrame.iloc[-testlength:]
DFrame = DFrame.iloc[:-testlength]

Tensor, DFrame = DataPrep.DataPreparation(p, DFrame)
print(str((len(DFrame) - sum(DFrame['none']))*2) + ' Training examples')
del DFrame

TestTensor, TestFrame = DataPrep.DataPreparation(p, TestFrame)

Model = TrainModel(p, Tensor)

TestPredictions = Model.predict(TestTensor)
# TradeSimulation.simulate(p, TestPredictions, test)

TestPredictions = pd.DataFrame(TestPredictions, index = TestFrame.index)
headers = [a+b for a, b in zip (p['TargetTickers'], ([' long'] * len(p['TargetTickers'])))]
headers.append('none')
TestPredictions.columns = headers

