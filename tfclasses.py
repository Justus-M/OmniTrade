import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from collections import Counter
from typing import Optional
from params import *
from time_series_data_handler import *


class TfData:
    data_frame: Optional[pd.DataFrame] = None
    params: Params
    handler: TimeSeriesDataHandler

    def __init__(self, handler):
        self.params = handler.params
        self.data_frame = handler.data_frame

    def window(self, tf_dataset):
        tf_dataset = tf_dataset.window(self.params.hindsight, 1, 1, True)

        if self.params.label_count != 0:
            tf_dataset = tf_dataset.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(self.params.hindsight), y)))
        else:
            tf_dataset = tf_dataset.flat_map(lambda x: tf.data.Dataset.zip(x.batch(self.params.hindsight)))

        return tf_dataset

    def batch_prefetch(self, tf_dataset):
        tf_dataset = tf_dataset.batch(self.params.batch_size).prefetch(16)
        return tf_dataset

    def time_series_tf_dataset(self, frame):
        tf_dataset = self.pandas_to_tf_dataset(frame)
        tf_dataset = self.window(tf_dataset)
        if self.params.filter_out_consecutive_signals:
            tf_dataset = tf_dataset.filter(lambda x, y: tf.math.equal(y[-1], 0))

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
        return self.time_series_tf_dataset(self.data_frame.iloc[:-self.validation_cutoff])

    @property
    def validation_data(self):
        return self.time_series_tf_dataset(self.data_frame.iloc[-self.validation_cutoff:-self.test_cutoff])

    @property
    def test_data(self):
        return self.time_series_tf_dataset(self.data_frame.iloc[-self.test_cutoff:])




