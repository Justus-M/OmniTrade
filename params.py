from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Purpose(Enum):
    training = 1
    live_prediction = 2


class ActivationFunction(Enum):
    softmax = 1
    sigmoid = 2


@dataclass()
class Params:
    _target_tickers: list = field(default_factory=lambda: ['SPY'])
    _tickers: list = field(default_factory=lambda: ['SPY'])
    # earliest year from which training data should start. By using 2010 the recession of 2007-2009 is excluded.
    year_cutoff: int = 2000
    # number of time interval periods behind used to explain price movements. Ex. with 5T (5 minute) interval,
    # we look at the last 256 5-minute intervals to predict the future.
    hindsight: int = 800
    # time period interval of data to use - in line with Pandas convention - T for minutes, D for days
    hindsight_interval: str = '30T'
    # number of time interval periods ahead for prediction.
    # ex. 32 with 5T interval means we look 32*5 = 160 minutes ahead
    foresight: int = 320
    # 0-0.99   minimum price increase for a buy signal.
    # Ex. 0.01 means we generate a buy signal for a price increase of at least 1% after the foresight period
    buy_threshold: float = 0.02
    # None or 0-0.99    minimum price decrease for a short signal
    sell_threshold: Optional[float] = None
    data_path: str = 'Data/Minute'
    validation_proportion: float = 0.2
    test_proportion: float = 0.1
    epochs: int = 20
    batch_size: int = 32
    bayesian_initial_points: int = 5
    bayesian_iterations: int = 20
    # Input Parameters above
    hindsight_extension: Optional[list] = None  # [1,2,3,4,8]
    purpose: Purpose = Purpose.training

    @property
    def displace(self) -> int:
        if self.sell_threshold is None:
            return 1
        else:
            return 2

    @property
    def activation(self) -> ActivationFunction:
        if self.sell_threshold is None and len(self.target_tickers) == 1:
            return ActivationFunction.softmax
        else:
            return ActivationFunction.sigmoid

    @property
    def tickers(self):
        new_tickers = list({*self._target_tickers, *self._tickers})
        new_tickers.sort()
        return new_tickers

    @property
    def target_tickers(self):
        new_tickers = self._target_tickers
        new_tickers.sort()
        return new_tickers

    @property
    def label_count(self): return 2

    @property
    def price_count(self): return 3



