import pandas as pd
import os
import ApiHelpers



tickers = pd.read_csv('SP500.csv')

for ticker in tickers['Symbol']:
    if not os.path.exists('Data/%s-Daily.csv' % (ticker)):
        ApiHelpers.get_daily_stocks(ticker, '4EHUONPLL0MA0NPU')
        temp = pd.read_csv('Data/%s-Daily.csv' % (ticker))
        if len(temp) < 5:
            tickers = tickers[tickers['Symbol'] != ticker]
            tickers.to_csv('SP500.csv')
            os.remove('Data/%s-Daily.csv' % (ticker))

for ticker in tickers['Symbol']:
    if not os.path.exists('Data/Adjusted/%s-Daily-Adjusted.csv' % (ticker)):
        ApiHelpers.get_daily_adjusted_stocks(ticker, '4EHUONPLL0MA0NPU')
        temp = pd.read_csv('Data/Adjusted/%s-Daily-Adjusted.csv' % (ticker))
        if len(temp) < 5:
            print('Alert')
            # tickers = tickers[tickers['Symbol'] != ticker]
            # tickers.to_csv('SP500.csv')
            os.remove('Data/Adjusted/%s-Daily-Adjusted.csv' % (ticker))


