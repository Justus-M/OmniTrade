def TsDataProcessor(Prices, *other, target = None, t=10):
    import pandas as pd
    import numpy as np
    from collections import deque

    frames = list([Prices])

    for i in other:
        frames.append(i)

    daterange = pd.DataFrame([])

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
    cols.remove(target + " close target")
    cols.append(target + " close target")
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

    if tickercol != None:
        for ticker in singles:
            if ticker != "APTV":
                for col in keep:
                    filtered[ticker + " " + col] = frame[frame[tickercol] == ticker][col]
                if ticker == "GOOG":
                    break

    filtered.dropna(inplace = True, axis = 1)

    if target != None:
        filtered[target + " close target"] = filtered[target + " close"].shift(periods=-time) / filtered[target + " close"]
        filtered.dropna(inplace=True)

    if time != None:
        filtered[target + " close target"] = (filtered[target + " close target"] > 1.00).astype(int)

    if target != None:
        for col in filtered.columns:
            if col != target + " close target":
                filtered[col] = filtered[col].pct_change()
                filtered.dropna(inplace=True)
            # filtered[col] = preprocessing.scale(filtered[col].values)
    else:
        for col in filtered.columns:
            filtered[col] = preprocessing.scale(filtered[col])

    filtered.dropna(inplace=True)

    return filtered


def split(data):
    import pandas
    import numpy as np

    X = []
    Y = []

    for x, y in data:
        X.append(x)
        Y.append(y)

    return np.array(X), Y

