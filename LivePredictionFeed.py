from TensorPrep import TensorPreparation
from Parameters import p
from matplotlib import pyplot
import time
import SP500Sync
import pytz
from ApiHelpers import get_minute_data
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model


MarketOpen = True #placeholder
ticker = p['TargetTickers'][0]
AlphaVantageKey = '4EHUONPLL0MA0NPU'
tz = pytz.timezone('America/New_York')
p['Purpose'] = 'LivePrediction'

if __name__ == '__main__':

    while MarketOpen:
        now = datetime.now(tz)
        EOD = datetime(now.year, now.month, now.day, 16, 00).replace(tzinfo=tz)

        LastOpen = SP500Sync.GetLastOpen(EOD, now)

        Model = load_model('models/1.h5')

        data = get_minute_data(ticker, AlphaVantageKey)

        columns = ticker + ' ' + frame.columns.values

        Tensor = TensorPreparation(p, data, RealizedPredictions = False, Balance = False)

        Prediction = Model.predict(Tensor)

        print(now)
        if Prediction[0][0]>Prediction[0][1]:
            print('Recommendation to buy ' + ticker)
        else:
            print('Recommendation not to buy ' + ticker)


        if LastOpen != now:
            print("Market Closed")
            break

        time.sleep(300)


