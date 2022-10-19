import pandas as pd
from sklearn import preprocessing
from params import *
import tensorflow as tf


class TimeSeriesDataHandler:
    data_frame: Optional[pd.DataFrame] = None
    params: Params

    def __init__(self, params):
        self.splits = ['training']
        self.params = params
        self.prepare_df()

    def prepare_df(self):
        for ticker in self.params.tickers:
            raw_frame = self.ingestion_and_preprocessing(ticker)
            self.append_data(raw_frame)
        self.feature_engineering()

    def ingestion_and_preprocessing(self, ticker: str):

        raw_frame = pd.read_csv(f'{str(self.params.data_path)}/{ticker}.csv', header=0, index_col='timestamp',
                                parse_dates=True)
        raw_frame.columns = ticker + ' ' + raw_frame.columns.values
        raw_frame.index = pd.to_datetime(raw_frame.index, errors='coerce')
        raw_frame = raw_frame.loc[pd.notnull(raw_frame.index)]

        raw_frame = raw_frame.resample(self.params.hindsight_interval).mean()
        raw_frame.dropna(inplace=True)
        raw_frame = raw_frame[raw_frame.index.year > self.params.year_cutoff]

        return raw_frame

    def append_data(self, raw_frame):
        # pull out into omni data class
        if self.data_frame is not None:
            self.data_frame = self.data_frame.merge(raw_frame, how='inner', left_index=True, right_index=True)
        else:
            self.data_frame = raw_frame

    def feature_engineering(self):

        if self.params.hindsight_extension is not None:
            self.extend_hindsight()

        self.create_target_variables()
        self.data_frame = self.data_frame.iloc[::-1]
        self.scale_x_variables()
        self.data_frame.dropna(inplace=True)

    def create_target_variables(self):
        target_price = pd.DataFrame(index=self.data_frame.index)
        price = pd.DataFrame(index=self.data_frame.index)
        self.data_frame['day'] = self.data_frame.index.day
        self.data_frame['month'] = self.data_frame.index.month
        self.data_frame['weekday'] = self.data_frame.index.weekday
        self.data_frame['minute'] = ((self.data_frame.index.hour - 9) * 60) + self.data_frame.index.minute - 30
        for ticker in self.params.target_tickers:
            target_price[ticker + ' upper_target'] = self.data_frame.apply(lambda x: self.data_frame.loc[x.name:].iloc[:self.params.foresight][ticker + ' close'].max(), axis=1)
            #target_price[ticker + ' lower_target'] = self.data_frame.apply(lambda x: self.data_frame.loc[x.name:].iloc[:self.params.foresight][ticker + ' close'].min(), axis=1)
            self.data_frame[ticker + ' max_return'] = ((target_price[ticker + ' upper_target'] / self.data_frame[ticker + ' close'])-1)*100
            #self.data_frame[ticker + ' max_drawdown'] = ((target_price[ticker + ' lower_target'] / self.data_frame[ticker + ' close'])-1)*100
            price[ticker + ' Price'] = self.data_frame[ticker + ' close']

        self.data_frame = self.data_frame.merge(price, how='inner', left_index=True, right_index=True)
        self.data_frame = self.data_frame.merge(target_price, how='inner', left_index=True, right_index=True)
        self.data_frame = self.data_frame.iloc[self.params.foresight:]
        self.data_frame.dropna(inplace=True)

    def scale_x_variables(self):
        col_names = self.data_frame.columns.values

        labels = self.data_frame[col_names[-(self.params.label_count + 2):]].copy()
        x_variables = self.data_frame[col_names[:-(self.params.label_count + 2)]].copy()
        for col in [c for c in x_variables.columns if any([c in a for a in ['open', 'high', 'low', 'close', 'volume']])]:
            x_variables[col] = x_variables[col]/x_variables[col].shift(periods=-1)
        x_variables.dropna(inplace=True)
        x_variables = x_variables.apply(lambda x: preprocessing.scale(x), axis=0)

        self.data_frame = x_variables.merge(labels, how='inner', left_index=True, right_index=True)

    def extend_hindsight(self):

        trail = pd.DataFrame(index=self.data_frame.index)

        open_hours = 6.5

        interval = self.params.hindsight_interval
        multiplier = 1
        if interval == '1D':
            multiplier = 1
        elif 'T' in interval:
            minutes = int(interval.replace('T', ''))
            per_hour = 60 / int(minutes)
            multiplier = per_hour * open_hours
        elif self.params.hindsight_interval == '1H':
            multiplier = open_hours
        elif self.params.hindsight_interval == '3H':
            multiplier = open_hours / 3.5

        for ticker in self.params.target_tickers:
            for weeks in self.params.hindsight_extension:
                trail[ticker + ' ' + str(self.params.hindsight_extension) + ' week hindsight'] = self.data_frame[
                    ticker + ' close'].shift(periods=-weeks * multiplier)

        self.data_frame = self.data_frame.merge(trail, how='inner', left_index=True, right_index=True)

    def window(self, tf_dataset, predict=False):
        shift = 1
        if not predict:
            shift = 4
        tf_dataset = tf_dataset.window(self.params.hindsight, shift, 1, True)

        if not predict:
            tf_dataset = tf_dataset.interleave(lambda x, y: tf.data.Dataset.zip((x.batch(self.params.hindsight), y)),
                                               num_parallel_calls=tf.data.AUTOTUNE).shuffle(100000)
        else:
            tf_dataset = tf_dataset.interleave(lambda x, _: tf.data.Dataset.zip(x.batch(self.params.hindsight)),
                                               num_parallel_calls=tf.data.AUTOTUNE
                                               )

        return tf_dataset.cache()

    def batch_prefetch(self, tf_dataset):
        tf_dataset = tf_dataset.batch(self.params.batch_size).prefetch(tf.data.AUTOTUNE)
        return tf_dataset

    def time_series_tf_dataset(self, frame, predict=False):
        tf_dataset = self.pandas_to_tf_dataset(frame)
        tf_dataset = self.window(tf_dataset, predict=predict)
        return self.batch_prefetch(tf_dataset)


    def pandas_to_tf_dataset(self, input_frame):

        columns = input_frame.columns.values
        if self.params.price_count > 0:
            input_frame = input_frame[columns[:-self.params.price_count]]

        if self.params.label_count > 0:
            columns = input_frame.columns.values
            variables = (tf.constant(input_frame[columns[:-self.params.label_count]].values),
                         tf.constant(input_frame[columns[-self.params.label_count:]].values))
        else:
            variables = (tf.constant(input_frame.values))
        tf_dataset = tf.data.Dataset.from_tensor_slices(variables)

        return tf_dataset

    @property
    def validation_cutoff(self):
        return int(len(self.data_frame)*(self.params.validation_proportion+self.params.test_proportion))

    @property
    def test_cutoff(self):
        return int(len(self.data_frame)*self.params.test_proportion)

    @property
    def training_data(self):
        return self.time_series_tf_dataset(self.data_frame.iloc[self.validation_cutoff:])

    @property
    def validation_data(self):
        return self.time_series_tf_dataset(self.data_frame.iloc[self.test_cutoff:self.validation_cutoff])

    @property
    def test_data(self):
        return self.time_series_tf_dataset(self.data_frame.iloc[:self.test_cutoff], predict=False)




