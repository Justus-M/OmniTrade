import pandas as pd
import TsDataProcessor

def ImportCrypto(x_names, y_name, TargetTickers, foresight, hindsight, buy_threshold, tickers, interval):
    frames = dict.fromkeys(tickers)

    baseindex = pd.read_csv("Data/" + tickers[0] + "-USD-train.csv", names=["time", "open", "close", "high", "low", "volume"],index_col = "time", parse_dates=True)
    baseindex.index = pd.to_datetime(baseindex.index, unit='ms')
    baseindex = baseindex.resample(interval).mean()

    All = pd.DataFrame(index = baseindex.index)
    del baseindex

    for ticker in tickers:
        frame = pd.read_csv("Data/" + ticker + "-USD-train.csv", names=["time", "open", "close", "high", "low", "volume"],index_col = "time", parse_dates=True)
        frame = frame[x_names] #only keep relevant columns
        frame.index = pd.to_datetime(frame.index, unit = 'ms')
        frame.columns = ticker + " " + frame.columns.values
        frame = frame.resample(interval).mean()
        All = All.merge(frame, how='inner', left_index=True, right_index=True)

    All = All.loc[All.index.drop_duplicates()]
    All.dropna(inplace=True)

    All["target"] = All[TargetTickers[0] + " " + y_name].shift(periods=-foresight) / All[TargetTickers[0] + " " + y_name]
    All["target"] = (All["target"] > 1 + buy_threshold).astype(int)

    TargetPrice = pd.DataFrame(index = All.index)
    Price = pd.DataFrame(index = All.index)

    for TargetTicker in TargetTickers:
        TargetPriceTemp = pd.DataFrame(All[TargetTicker + " " + y_name].shift(periods=-foresight)).copy()
        TargetPriceTemp.columns.values[0] = TargetTicker + " TargetPrice"
        TargetPrice = TargetPrice.merge(TargetPriceTemp, how='inner', left_index=True, right_index=True)
        PriceTemp = pd.DataFrame(All[TargetTicker + " " + y_name]).copy()
        PriceTemp.columns.values[0] = TargetTicker + " Price"
        Price = Price.merge(PriceTemp, how='inner', left_index=True, right_index=True)

    TargetPrice.dropna(inplace=True)
    Price.dropna(inplace=True)


    trades = pd.read_csv("Data/" + TargetTickers[0] + "-USD-trades.csv", names=["ID", "time", "AMOUNT", "PRICE"],index_col = "time", parse_dates=True)
    trades.index = pd.to_datetime(trades.index, unit = 'ms')
    trades = trades.resample(interval).sum()
    All = All.merge(trades["AMOUNT"], how ='inner', left_index=True, right_index=True)
    All.dropna(inplace=True)

    del trades

    All = TsDataProcessor.Scaler(All, "target")
    All.dropna(inplace=True)


    return All[::-1], TargetPrice[::-1], Price[::-1]

