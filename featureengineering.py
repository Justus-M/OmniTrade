import pandas as pd
from sklearn import preprocessing
import numpy as np
from params import *


def ExtendHindsight(frame, params: Params):

    trail = pd.DataFrame(index=frame.index)

    open_hours = 6.5

    interval = params.hindsight_interval
    multiplier = 1
    if interval == '1D':
        multiplier = 1
    elif 'T' in interval:
        minutes = int(interval.replace('T',''))
        perhour = 60/int(minutes)
        multiplier = perhour*open_hours
    elif params.hindsight_interval == '1H':
        multiplier = open_hours
    elif params.hindsight_interval == '3H':
        multiplier = open_hours/3.5

    for ticker in params.target_tickers:
        for weeks in params.hindsight_extension:
            trail[ticker + ' ' + str(params.hindsight_extension) + ' week hindsight'] = frame[ticker + ' close'].shift(periods=-weeks * multiplier)

    return trail


def scaler(frame):

    for col in frame.columns.values:
        frame[col] = preprocessing.scale(frame[col].values)

    frame.dropna(inplace=True)

    return frame


def feature_engineering(params: Params, input_frame, realized_predictions=True):

    if params.hindsight_extension is not None:
        trail = ExtendHindsight(input_frame, params)
        input_frame = input_frame.merge(trail, how='inner', left_index=True, right_index=True)

    if realized_predictions:

        target_price = pd.DataFrame(index=input_frame.index)
        price = pd.DataFrame(index=input_frame.index)

        for ticker in params.target_tickers:
            target_price[ticker + ' TargetPrice'] = input_frame[ticker + ' close'].shift(periods=params.foresight)
            stock_return = target_price[ticker + ' TargetPrice'] / input_frame[ticker + ' close']
            stock_return.dropna(inplace=True)
            input_frame[ticker + ' long'] = (stock_return > 1 + params.buy_threshold).astype(int)
            price[ticker + ' Price'] = input_frame[ticker + ' close']
            if params.sell_threshold is not None:
                input_frame[ticker + ' short'] = (stock_return < 1 - params.sell_threshold).astype(int)

        ColNames = input_frame.columns.values

        input_frame['none'] = (input_frame[ColNames[-(params.label_count - 1):]].sum(axis = 1) == 0).astype(int)

        ColNames = input_frame.columns.values

        labels = input_frame[ColNames[-params.label_count:]].copy()
        x_variables = input_frame[ColNames[:-params.label_count]].copy()
        del input_frame

        x_variables = scaler(x_variables)

        input_frame = x_variables.merge(labels, how='inner', left_index=True, right_index=True)
        del x_variables
        input_frame = input_frame.merge(price, how='inner', left_index=True, right_index=True)
        input_frame = input_frame.merge(target_price, how='inner', left_index=True, right_index=True)
        del price
        del target_price

        input_frame.dropna(inplace=True)

    else:
        input_frame = scaler(input_frame)

    return input_frame


def feature_engineering_stack(params: Params, input_frame, ticker, realized_predictions=True):

    if params.hindsight_extension is not None:
        trail = ExtendHindsight(input_frame, params)
        input_frame = input_frame.merge(trail, how='inner', left_index=True, right_index=True)

    if realized_predictions:
        target_price = pd.DataFrame(index=input_frame.index)
        price = pd.DataFrame(index=input_frame.index)

        target_price[ticker + ' TargetPrice'] = input_frame[ticker + ' close'].shift(periods=params.foresight)
        stock_return = target_price[ticker + ' TargetPrice'] / input_frame[ticker + ' close']
        stock_return.dropna(inplace=True)
        input_frame[ticker + ' long'] = (stock_return > 1 + params.buy_threshold).astype(int)
        price[ticker + ' Price'] = input_frame[ticker + ' close']
        if params.sell_threshold is not None:
            input_frame[ticker + ' short'] = (stock_return < 1 - params.sell_threshold).astype(int)


        ColNames = input_frame.columns.values

        input_frame['none'] = (input_frame[ColNames[-(params.label_count - 1):]].sum(axis = 1) == 0).astype(int)

        ColNames = input_frame.columns.values

        labels = input_frame[ColNames[-params.label_count:]].copy()
        print(len(labels)-sum(labels['none']))
        x_variables = input_frame[ColNames[:-params.label_count]].copy()
        del input_frame

        # x_variables = (x_variables.shift(periods=1) / x_variables)
        x_variables.replace([np.inf, -np.inf], np.nan, inplace=True)
        x_variables.dropna(inplace=True)
        x_variables = scaler(x_variables)

        input_frame = x_variables.merge(labels, how='inner', left_index=True, right_index=True)
        del x_variables
        input_frame = input_frame.merge(price, how='inner', left_index=True, right_index=True)
        input_frame = input_frame.merge(target_price, how='inner', left_index=True, right_index=True)
        del price
        del target_price

        input_frame.dropna(inplace=True)

    else:
        input_frame = scaler(input_frame)


    return input_frame

