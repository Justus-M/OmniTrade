from time_series_data_handler import TimeSeriesDataHandler
from modeling import *
from params import *

## Verify target variables are calculated correctly in raw df

data = TimeSeriesDataHandler(Params())
model = MarketForecastNN()
model.train_model(data)
predictions = model.model.predict(data.test_data)

class OmniTrade:
    def __init__(self):
        self.data = TimeSeriesDataHandler(Params())
        self.model = MarketForecastNN()

    def get_test_predictions(self):
        predictions = self.model.model.predict(self.data.test_data)
        self.test_frame = self.data.data_frame[self.data.test_cutoff:].iloc[:len(predictions):-1]
        self.test_frame[['predicted spike', 'predicted drawdown']] = predictions


