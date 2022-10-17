from time_series_data_handler import TimeSeriesDataHandler
from modeling import *
from params import *

## Verify target variables are calculated correctly in raw df

class OmniTrade:
    def __init__(self):
        self.data = TimeSeriesDataHandler(Params())
        self.model = MarketForecastNN()
        self.model.train_model(self.data)
    def get_test_predictions(self):
        predictions = self.model.model.predict(self.data.test_data)
        self.test_frame = self.data.data_frame[self.data.test_cutoff:].iloc[1:len(predictions)]
        self.test_frame[['predicted spike', 'predicted drawdown']] = predictions

    def get_top_signals(self):
        self.long_signals = self.test_frame.sort_values(by='predicted spike', ascending=False)
        filter_list = [abs(self.long_signals.index[i+1]-self.long_signals.index[i])>2 for i in range(len(self.long_signals)-1)]
        self.long_signals = self.long_signals.iloc[:-1][filter_list].iloc[:20]
        self.short_signals = self.test_frame.sort_values(by='predicted drawdown', ascending=False)
        filter_list = [abs(self.short_signals.index[i + 1] - self.short_signals.index[i]) > 2 for i in
                       range(len(self.short_signals) - 1)]
        self.short_signals = self.short_signals.iloc[:-1][filter_list].iloc[:20]




t = OmniTrade()
frame = t.get_test_predictions()




