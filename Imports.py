import pandas as pd
import TsDataProcessor

def ImportTimeseries(p):
    frames = dict.fromkeys(p.tickers)

    baseindex = pd.read_csv("Data/" + p.tickers[0] + "-USD-train.csv", names=["time", "open", "close", "high", "low", "volume"],index_col = "time", parse_dates=True)
    baseindex.index = pd.to_datetime(baseindex.index, unit='ms')
    baseindex = baseindex.resample(p.hindsight_interval).mean()

    All = pd.DataFrame(index = baseindex.index)
    del baseindex

    for ticker in p.tickers:
        frame = pd.read_csv("Data/" + ticker + "-USD-train.csv", names=["time", "open", "close", "high", "low", "volume"],index_col = "time", parse_dates=True)
        frame = frame[p.x_names] #only keep relevant columns
        frame.index = pd.to_datetime(frame.index, unit = 'ms')
        frame.columns = ticker + " " + frame.columns.values
        frame = frame.resample(p.hindsight_interval).mean()
        All = All.merge(frame, how='inner', left_index=True, right_index=True)

    All = All.loc[All.index.drop_duplicates()]
    All.dropna(inplace=True)

    trades = pd.read_csv("Data/" + p.TargetTickers[0] + "-USD-trades.csv", names=["ID", "time", "AMOUNT", "PRICE"],
                         index_col="time", parse_dates=True)
    trades.index = pd.to_datetime(trades.index, unit='ms')
    trades = trades.resample(p.hindsight_interval).sum()
    All = All.merge(trades["AMOUNT"], how='inner', left_index=True, right_index=True)
    All.dropna(inplace=True)

    del trades

    All["target"] = All[p.TargetTickers[0] + " " + p.y_name].shift(periods=-p.foresight) / All[p.TargetTickers[0] + " " + p.y_name]
    All["long"] = (All["target"] > 1 + p.buy_threshold).astype(int)
    if p.sell_threshold != None:
        All["short"] = (All["target"] > 1 - p.sell_threshold).astype(int)
        All["none"] = ((All["target"] < 1 + p.buy_threshold) & (All["target"] > 1 - p.sell_threshold)).astype(int)
    All = All.drop(["target"], axis = 1)

    TargetPrice = pd.DataFrame(index = All.index)
    Price = pd.DataFrame(index = All.index)

    for TargetTicker in p.TargetTickers:
        TargetPriceTemp = pd.DataFrame(All[TargetTicker + " " + p.y_name].shift(periods=-p.foresight)).copy()
        TargetPriceTemp.columns.values[0] = TargetTicker + " TargetPrice"
        TargetPrice = TargetPrice.merge(TargetPriceTemp, how='inner', left_index=True, right_index=True)
        PriceTemp = pd.DataFrame(All[TargetTicker + " " + p.y_name]).copy()
        PriceTemp.columns.values[0] = TargetTicker + " Price"
        Price = Price.merge(PriceTemp, how='inner', left_index=True, right_index=True)

    TargetPrice.dropna(inplace=True)
    Price.dropna(inplace=True)

    displace = 0
    if p.sell_threshold == None:
        displace = 2

    ScaledFeatures = TsDataProcessor.Scaler(All[All.columns.values[:-3+displace]])
    Labels = All[All.columns.values[-3+displace:]]
    All = ScaledFeatures.merge(Labels, how='inner', left_index=True, right_index=True)
    All.dropna(inplace=True)


    return All[::-1], TargetPrice[::-1], Price[::-1]

