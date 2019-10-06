import time
import pandas as pd
import tensorflow as tf
import DataPrep
from sklearn import preprocessing

def Split(Frame, Proportion):

    Child_len = int(Proportion * len(Frame))
    Childframe = Frame.iloc[:Child_len].copy()
    Frame = Frame.iloc[Child_len:]

    return Frame, Childframe

def ExtendHindsight(frame, p):

    Trail = pd.DataFrame(index = frame.index)

    OpenHours = 6.5

    interval = p['hindsight_interval']

    if interval == '1D':
        multiplier = 1
    elif 'T' in interval:
        minutes = int(interval.replace('T',''))
        perhour = 60/int(minutes)
        multiplier = perhour*OpenHours
    elif p['hindsight_interval'] == '1H':
        multiplier = OpenHours
    elif p['hindsight_interval'] == '3H':
        multiplier = OpenHours/3.5

    for ticker in p['TargetTickers']:
        for weeks in p['HindsightExtension']:
            Trail[ticker + ' ' + str(p['HindsightExtension']) + ' week hindsight'] = frame[ticker + ' ' + p['y_name']].shift(periods=-weeks*multiplier)

    return Trail

def Scaler(frame):

    for col in frame.columns.values:
        frame[col] = preprocessing.scale(frame[col].values)

    frame.dropna(inplace=True)

    return frame

def Tensify(dataframe, p, Y = True):

    if Y:
        variables = (tf.constant(dataframe[dataframe.columns.values[:-p['LabelCount']]].values), tf.constant(dataframe[dataframe.columns.values[-p['LabelCount']:]].values))
    else:
        variables = (tf.constant(dataframe.values))
    tensor = tf.data.Dataset.from_tensor_slices(variables)
    tensor = tensor.window(p['hindsight'],1,1,True)
    if Y:
        tensor = tensor.flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(p['hindsight']), y)))
    else:
        tensor = tensor.flat_map(lambda x: tf.data.Dataset.zip(x.batch(p['hindsight'])))
    return tensor

def BalanceTensor(tensor, npos, reduce):

    positive = tensor.filter(lambda x,y: tf.math.equal(y[-1],0))
    negative = tensor.filter(lambda x,y: tf.math.equal(y[-1],1))

    if reduce == 'Negative':
        negative = negative.shuffle(20000)
    else:
        positive = positive.shuffle(20000)

    negative = negative.take(npos)
    positive = positive.take(npos)
    tensor = positive.concatenate(negative)
    tensor = tensor.shuffle(20000)
    return tensor

def TensorPreparation(p, DFrame, RealizedPredictions = True, Balance = True):

    then = time.time()

    if not p['HindsightExtension'] == None:
        Trail = ExtendHindsight(DFrame, p)
        DFrame = DFrame.merge(Trail, how='inner', left_index=True, right_index=True)
    print(str(time.time() - then) + ' 1')

    if RealizedPredictions:

        TargetPrice = pd.DataFrame(index=DFrame.index)
        Price = pd.DataFrame(index=DFrame.index)

        for ticker in p['TargetTickers']:
            TargetPrice[ticker + ' TargetPrice'] = DFrame[ticker + ' ' + p['y_name']].shift(periods=p['foresight'])
            Return = TargetPrice[ticker + ' TargetPrice'] / DFrame[ticker + ' ' + p['y_name']]
            Return.dropna(inplace = True)
            DFrame[ticker + ' long'] = (Return > 1 + p['buy_threshold']).astype(int)
            Price[ticker + ' Price'] = DFrame[ticker + ' ' + p['y_name']]
            if p['sell_threshold'] != None:
                DFrame[ticker +' short'] = (Return < 1 - p['sell_threshold']).astype(int)

        print(str(time.time() - then) + ' 2')

        DFrame['none'] = (DFrame[DFrame.columns.values[-(p['LabelCount']-1):]].sum(axis = 1) == 0).astype(int)

        print(str(time.time() - then) + ' 3')

        Labels = DFrame[DFrame.columns.values[-p['LabelCount']:]].copy()
        Xvariables = DFrame[DFrame.columns.values[:-p['LabelCount']]].copy()
        del DFrame

        Xvariables = Scaler(Xvariables)
        print(str(time.time() - then) + ' 4')

        DFrame = Xvariables.merge(Labels, how='inner', left_index=True, right_index=True)
        del Xvariables
        DFrame = DFrame.merge(Price, how='inner', left_index=True, right_index=True)
        DFrame = DFrame.merge(TargetPrice, how='inner', left_index=True, right_index=True)
        del Price
        del TargetPrice

        print(str(time.time() - then) + ' 5')

        DFrame.dropna(inplace=True)

        tensor = Tensify(DFrame[DFrame.columns.values[:-(len(p['TargetTickers']) * 2)]], p)

        print(str(time.time() - then) + ' 6')

        if Balance:
            lowest = min(len(DFrame) - sum(DFrame['none']), sum(DFrame['none']))

            if lowest == sum(DFrame['none']):
                reduce = 'Negative'
            else:
                reduce = 'Positive'

            tensor = BalanceTensor(tensor, lowest, reduce)
    else:
        DFrame = Scaler(DFrame)
        tensor = Tensify(DFrame, p, Y = False)

    tensor = tensor.batch(p['Batch_size']).prefetch(1)

    print(str(time.time() - then) + ' 7')

    DFrame = DFrame.iloc[:-p['hindsight']+1]

    return tensor, DFrame

