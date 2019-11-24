import time
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

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
            Trail[ticker + ' ' + str(p['HindsightExtension']) + ' week hindsight'] = frame[ticker + ' close'].shift(periods=-weeks*multiplier)

    return Trail

def Scaler(frame):

    for col in frame.columns.values:
        frame[col] = preprocessing.scale(frame[col].values)

    frame.dropna(inplace=True)

    return frame

def TensorDatasetPreparation(p, DFrame, RealizedPredictions = True, Balance = True):

    if not p['HindsightExtension'] == None:
        Trail = ExtendHindsight(DFrame, p)
        DFrame = DFrame.merge(Trail, how='inner', left_index=True, right_index=True)

    if RealizedPredictions:

        TargetPrice = pd.DataFrame(index=DFrame.index)
        Price = pd.DataFrame(index=DFrame.index)

        for ticker in p['TargetTickers']:
            TargetPrice[ticker + ' TargetPrice'] = DFrame[ticker + ' close'].shift(periods=p['foresight'])
            Return = TargetPrice[ticker + ' TargetPrice'] / DFrame[ticker + ' close']
            Return.dropna(inplace = True)
            DFrame[ticker + ' long'] = (Return > 1 + p['buy_threshold']).astype(int)
            Price[ticker + ' Price'] = DFrame[ticker + ' close']
            if p['sell_threshold'] != None:
                DFrame[ticker +' short'] = (Return < 1 - p['sell_threshold']).astype(int)


        ColNames = DFrame.columns.values

        DFrame['none'] = (DFrame[ColNames[-(p['LabelCount']-1):]].sum(axis = 1) == 0).astype(int)

        ColNames = DFrame.columns.values

        Labels = DFrame[ColNames[-p['LabelCount']:]].copy()
        Xvariables = DFrame[ColNames[:-p['LabelCount']]].copy()
        del DFrame

        Xvariables = Scaler(Xvariables)

        DFrame = Xvariables.merge(Labels, how='inner', left_index=True, right_index=True)
        del Xvariables
        DFrame = DFrame.merge(Price, how='inner', left_index=True, right_index=True)
        DFrame = DFrame.merge(TargetPrice, how='inner', left_index=True, right_index=True)
        del Price
        del TargetPrice

        DFrame.dropna(inplace=True)

    else:
        DFrame = Scaler(DFrame)


    return DFrame

