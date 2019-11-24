import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from tensorflow import keras
from bayes_opt import BayesianOptimization as BayesOpt
from tensorflow.keras.models import Sequential


class tfData():

    def PandasImport(self, path = None, index = None, parse_dates = False):

        self.TrainingFrame = pd.read_csv(path, index_col = index, parse_dates = parse_dates)

    def Scaler(self, Frame):

        for col in Frame.columns.values:
            Frame[col] = preprocessing.scale(Frame[col].values)

        Frame.dropna(inplace=True)

        return Frame

    def TensorDataset(self, Dataframe, LabelCount = 0, Exclude = 0):

        Columns = Dataframe.columns.values
        Dataframe = Dataframe[Columns[:-Exclude]]

        if LabelCount > 0:
            Columns = Dataframe.columns.values

            variables = (tf.constant(Dataframe[Columns[:-LabelCount]].values),
                         tf.constant(Dataframe[Columns[-LabelCount:]].values))
        else:
            variables = (tf.constant(Dataframe.values))
        TensorDataset = tf.data.Dataset.from_tensor_slices(variables)

        return TensorDataset

    def _Split(self, Proportion):

        ProportionLength = int(Proportion * len(self.TrainingFrame))
        SplitDFrame = self.TrainingFrame.iloc[:ProportionLength].copy()
        self.TrainingFrame = self.TrainingFrame.iloc[ProportionLength:]

        return SplitDFrame

    def Split(self, Validation = 0.2, Test = 0):

        self.Splits = ['Training']

        if Validation > 0:
            self.ValidationFrame = self._Split(Validation)
            self.Splits.append('Validation')
        if Test > 0:
            self.TestFrame = self._Split(Test)
            self.Splits.append('Training')

    def Window(self, TensorDataset, WindowSize, Labels = True):

        TensorDataset = TensorDataset.window(WindowSize, 1, 1, True)

        if Labels:
            TensorDataset = TensorDataset.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(WindowSize), y)))
        else:
            TensorDataset = TensorDataset.flat_map(lambda x: tf.data.Dataset.zip(x.batch(WindowSize)))

        return TensorDataset

    def BinaryClassBalance(self, Dataframe, TensorDataset):

        PositiveSum = sum(Dataframe[Dataframe.columns.values[-2]])
        Less = int(min(len(Dataframe) - PositiveSum, PositiveSum))

        one = TensorDataset.filter(lambda x, y: tf.math.equal(y[-1], 0))
        zero = TensorDataset.filter(lambda x, y: tf.math.equal(y[-1], 1))

        if PositiveSum<len(Dataframe) - PositiveSum:
            zero = zero.shuffle(20000)
            one = one.shuffle(20000)
        else:
            zero = zero.shuffle(20000)
            one = one.shuffle(20000)

        zero = zero.take(Less)
        one = one.take(Less)
        TensorDataset = one.concatenate(zero)
        TensorDataset = TensorDataset.shuffle(20000)

        return TensorDataset

    def BatchPrefetch(self, TensorDataset, Batchsize):
        TensorDataset = TensorDataset.batch(Batchsize).prefetch(1)
        return TensorDataset

    def TsTensorDataset(self, LabelCount = 1, Exclude = 0, WindowSize = 128, Batchsize = 1):

        for Part in self.Splits:
            Dataframe = getattr(self, '%sFrame' % Part)
            TensorDataset = self.TensorDataset(Dataframe, LabelCount = LabelCount, Exclude = Exclude)
            TensorDataset = self.Window(TensorDataset, WindowSize = WindowSize)
            if Part != 'Test': TensorDataset = self.BinaryClassBalance(Dataframe[Dataframe.columns.values[:-Exclude]], TensorDataset)
            TensorDataset = self.BatchPrefetch(TensorDataset, Batchsize)
            Dataframe = Dataframe.iloc[:-WindowSize + 1]
            setattr(self, 'Tf%sDataset' % Part, TensorDataset)
            setattr(self, '%sFrame' % Part, Dataframe.iloc[:-WindowSize+1])

class KerasModel:

    def __init__(self):
        self.Model = keras.models.Sequential()

    def Model(self):
        self.Model = Sequential()


    def BayesianOptimization(self, **Parameters):

        pbounds =  {}

        for Parameter, Value in Parameters.items():
            pbounds[Parameter] = Value

        optimizer = BayesOpt(
            f=OptimizationEvaluation,
            pbounds=pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        optimizer.maximize(init_points=10, n_iter=100)
        print(optimizer.max)










