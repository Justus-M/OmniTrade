def StockFilter(frame, tickercol = None):

    import pandas as pd

    singles = list(dict.fromkeys(frame[tickercol]))
    filtered = pd.DataFrame(index=frame[frame[tickercol] == singles[0]].index)

    for ticker in singles:
        for col in filtered.columns.values:
            filtered[ticker + " " + col] = frame[frame[tickercol] == ticker][col]

    filtered.dropna(inplace = True, axis = 1)

    return filtered

def Scaler(frame):
    import numpy as np

    for col in frame.columns.values:
        if col.endswith("AMOUNT"):
            frame[col] = preprocessing.scale(frame[col].values)
        else:
            frame[col] = frame[col].pct_change()
            frame = frame.replace([np.inf, -np.inf], np.nan)
            frame.dropna(inplace=True)
            frame[col] = preprocessing.scale(frame[col].values)

    frame.dropna(inplace=True)

    return frame

def Tensify(dataframe, p, Y = True):
    if Y:
        if p.sell_threshold == None and len(p.TargetTickers) == 1:
            variables = (tf.constant(dataframe[dataframe.columns.values[:-1]].values), tf.constant(dataframe[dataframe.columns.values[-1]].values))
        else:
            variables = (tf.constant(dataframe[dataframe.columns.values[:-p.LabelCount]].values), tf.constant(dataframe[dataframe.columns.values[-p.LabelCount:]].values))
    else:
        variables = (tf.constant(dataframe.values))
    tensor = tf.data.Dataset.from_tensor_slices(variables)
    tensor = tensor.window(p.hindsight,1,1,True)
    if Y:
        if p.sell_threshold == None and len(p.TargetTickers) == 1:
            tensor = tensor.flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(p.hindsight), y.batch(1))))
        else:
            tensor = tensor.flat_map(lambda x,y: tf.data.Dataset.zip((x.batch(p.hindsight), y)))
    else:
        tensor = tensor.flat_map(lambda x: tf.data.Dataset.zip(x.batch(p.hindsight)))
    return tensor

def BalanceTensor(tensor, npos, p):

    if p.sell_threshold != None and len(p.TargetTickers) == 1:
        positive = tensor.filter(lambda x,y: tf.math.equal(y[-1],0))
        negative = tensor.filter(lambda x,y: tf.math.equal(y[-1],1))
    else:
        positive = tensor.filter(lambda x,y: tf.math.equal(y[-1],1))
        negative = tensor.filter(lambda x,y: tf.math.equal(y[-1],0))
    negative = negative.shuffle(20000)
    negative = negative.take(npos)
    tensor = positive.concatenate(negative)
    tensor = tensor.shuffle(20000)
    return tensor


