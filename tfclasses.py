import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from collections import Counter


class tf_data():

    def __init__(self):
        self.splits = ['training']


    def pandas_import(self, path = None, index = None, parse_dates = False):

        self.training_frame = pd.read_csv(path, index_col = index, parse_dates = parse_dates)

    def scaler(self, frame):

        for col in frame.columns.values:
            frame[col] = preprocessing.scale(frame[col].values)

        frame.dropna(inplace=True)

        return frame

    def pandas_to_tf_dataset(self, input_frame, label_count = 0, cols_exclude = 0):

        columns = input_frame.columns.values
        if cols_exclude > 0:
            input_frame = input_frame[columns[:-cols_exclude]]

        if label_count > 0:
            columns = input_frame.columns.values

            variables = (tf.constant(input_frame[columns[:-label_count]].values),
                         tf.constant(input_frame[columns[-label_count:]].values))
        else:
            variables = (tf.constant(input_frame.values))
        tf_dataset = tf.data.Dataset.from_tensor_slices(variables)

        return tf_dataset

    def _split(self, proportion):

        split_length = int(proportion * len(self.training_frame))
        split_frame = self.training_frame.iloc[:split_length].copy()
        self.training_frame = self.training_frame.iloc[split_length:]

        return split_frame

    def split(self, validation = 0.2, test = 0.0):

        if validation > 0:
            self.validation_frame = self._split(validation)
            self.splits.append('validation')
        if test > 0:
            self.test_frame = self._split(test)
            self.splits.append('test')

    def window(self, tf_dataset, window_size, labels = True):

        tf_dataset = tf_dataset.window(window_size, 1, 1, True)

        if labels:
            tf_dataset = tf_dataset.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(window_size), y)))
        else:
            tf_dataset = tf_dataset.flat_map(lambda x: tf.data.Dataset.zip(x.batch(window_size)))

        return tf_dataset

    def binary_class_balance(self, frame, tf_dataset):

        label_count = Counter(frame[frame.columns.values[-1]])
        positive_count = label_count[0]
        negative_count = label_count[1]

        Less = min(negative_count, positive_count)

        positives = tf_dataset.filter(lambda x, y: tf.math.equal(y[-1], 0))
        negatives = tf_dataset.filter(lambda x, y: tf.math.equal(y[-1], 1))

        if positive_count<len(frame) - positive_count:
            negatives = negatives.shuffle(20000)
            positives = positives.shuffle(20000)
        else:
            negatives = negatives.shuffle(20000)
            positives = positives.shuffle(20000)

        negatives = negatives.take(Less)
        positives = positives.take(Less)
        tf_dataset = positives.concatenate(negatives)
        tf_dataset = tf_dataset.shuffle(20000)

        return tf_dataset

    def batch_prefetch(self, tf_dataset, batch_size):
        tf_dataset = tf_dataset.batch(batch_size).prefetch(16)
        return tf_dataset

    def time_series_tf_dataset(self, label_count = 0, cols_exclude = 0, window_size = 128, batch_size = 1):

        for part in self.splits:
            frame = getattr(self, f'{part}_frame')
            tf_dataset = self.pandas_to_tf_dataset(frame, label_count = label_count, cols_exclude = cols_exclude)
            tf_dataset = self.window(tf_dataset, window_size = window_size, labels = label_count!=0)
            frame = frame.iloc[:-window_size + 1]
            setattr(self, f'{part}_frame', frame.iloc[:-window_size + 1])
            setattr(self, f'tf_{part}_dataset', tf_dataset)

        if self.folds and self.folds[0] in list(range(self.folds[1]))[1:-1]:
            self.training_frame = pd.concat([self.training_frame, self.test_frame])
            self.tf_training_dataset = self.tf_training_dataset.concatenate(self.tf_test_dataset)


        for part in self.splits:
            if part != 'test' and label_count!=0:
                frame = getattr(self, f'{part}_frame')
                tf_dataset = getattr(self, f'tf_{part}_dataset')
        #        tf_dataset = self.binary_class_balance(frame[frame.columns.values[:-cols_exclude]], tf_dataset)
                setattr(self, f'tf_{part}_dataset', tf_dataset)

        for part in self.splits:
            tf_dataset = getattr(self, f'tf_{part}_dataset')
            tf_dataset = self.batch_prefetch(tf_dataset, batch_size)
            setattr(self, f'tf_{part}_dataset', tf_dataset)

    def rename_pred(self, old_name = 'training_frame', new_name = 'pred_frame'):

        self.__dict__[new_name] = self.__dict__.pop(old_name)
        self.splits = ['pred']










