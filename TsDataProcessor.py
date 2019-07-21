def TsDataProcessor(Base, *other, target = None, t=10):
    import pandas as pd
    import numpy as np
    from collections import deque

    frames = list([Base])

    for i in other:
        frames.append(i)

    for i in range(len(frames) - 1, -1, -1):
        if type(frames[i]) != pd.Series and type(frames[i]) != pd.DataFrame:
            print("Only data frames and series may be submitted as arguments for *other, not: " + str(type(frames[i])))
            if type(frames[i]) == pd.Series:
                frames[i] = frames[i].to_frame()

    dataset = frames[0]

    for i in range(1, len(frames)):
        for col in frames[i].columns.values:
            dataset[col] = frames[i][col]

    dataset.dropna(inplace=True)

    # check for NaN in date range and report to report tainted data quality

    # for i in range(0, 5):
    #     dataset["day" + str(i)] = (dataset.index.dayofweek == i).astype(int)

    cols = list(dataset.columns)
    cols.remove(target)
    cols.append(target)
    dataset = dataset[cols]

    sequential_data = []
    prev_days = deque(maxlen=t)

    for i in dataset.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == t:
            sequential_data.append([np.array(prev_days), i[-1]])

    return dataset, sequential_data


def StockFilter(frame, *keep, tickercol = None, target = None, time = None):

    import pandas as pd
    from sklearn import preprocessing
    singles = list(dict.fromkeys(frame[tickercol]))
    filtered = pd.DataFrame(index=frame[frame[tickercol] == singles[0]].index)

    for ticker in singles:
        for col in filtered.columns.values:
            filtered[ticker + " " + col] = frame[frame[tickercol] == ticker][col]

    filtered.dropna(inplace = True, axis = 1)

def Scaler(frame, target):
    from sklearn import preprocessing

    for col in frame.columns.values:
        if col != target:
            frame[col] = frame[col].pct_change()
            frame.dropna(inplace=True)
            frame[col] = preprocessing.scale(frame[col].values)

    frame.dropna(inplace=True)

    return frame

def Balance(data):

    import random

    buys = []
    sells = []

    for seq, target in data:
        if target == 1:
            buys.append([seq, target])
        else:
            sells.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    data = buys[:lower] + sells[:lower]

    random.shuffle(data)

    return data

def split(data):
    import numpy as np

    X = []
    Y = []

    for x, y in data:
        X.append(x)
        Y.append(y)

    return np.array(X), Y

