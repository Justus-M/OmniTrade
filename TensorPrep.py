import time
import pandas as pd
import importlib
import tensorflow as tf
import random
import DataPrep
import DataPrep
import torch
import os
import numpy as np
#import plaidml
import numpy as np
from sklearn import preprocessing

importlib.reload(DataPrep)

def Split(Frame, Proportion):

    Child_len = int(Proportion * len(Frame))
    Childframe = Frame.iloc[:Child_len]
    Frame = Frame.iloc[Child_len:]

    return Frame, Childframe

def Scaler(frame):

    for col in frame.columns.values:
        if col.endswith('AMOUNT'):
            frame[col] = preprocessing.scale(frame[col].values)
        else:
            # frame[col] = frame[col].pct_change()
            frame = frame.replace([np.inf, -np.inf], np.nan)
            frame.dropna(inplace=True)
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

def BalanceTensor(tensor, npos, p):

    positive = tensor.filter(lambda x,y: tf.math.equal(y[-1],0))
    negative = tensor.filter(lambda x,y: tf.math.equal(y[-1],1))
    negative = negative.shuffle(20000)
    negative = negative.take(npos)
    positive = positive.shuffle(20000)
    positive = positive.take(npos)
    tensor = positive.concatenate(negative)
    tensor = tensor.shuffle(20000)
    return tensor

def TensorPreparation(p, DFrame, RealizedPredictions = True, Balance = True):

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

        DFrame['none'] = (DFrame[DFrame.columns.values[-(p['LabelCount']-1):]].sum(axis = 1) == 0).astype(int)

        ScaledFeatures = Scaler(DFrame[DFrame.columns.values[:-p['LabelCount']]])
        Labels = DFrame[DFrame.columns.values[-p['LabelCount']:]]

        DFrame = ScaledFeatures.merge(Labels, how='inner', left_index=True, right_index=True)
        DFrame = DFrame.merge(Price, how='inner', left_index=True, right_index=True)
        DFrame = DFrame.merge(TargetPrice, how='inner', left_index=True, right_index=True)

        DFrame.dropna(inplace=True)

        tensor = Tensify(DFrame[DFrame.columns.values[:-(len(p['TargetTickers']) * 2)]], p)

        if Balance:
            tensor = BalanceTensor(tensor, min(len(DFrame) - sum(DFrame['none']), sum(DFrame['none'])), p)
    else:
        DFrame = Scaler(DFrame)
        tensor = Tensify(DFrame, p, Y = False)

    tensor = tensor.batch(p['Batch_size']).prefetch(1)

    DFrame = DFrame.iloc[:-p['hindsight']+1]

    return tensor, DFrame

