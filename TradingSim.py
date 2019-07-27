from datetime import datetime
cryptos = dict.fromkeys(["BTC", "ETH", "LTC"])

for i in cryptos:
    cryptos[i] = pd.read_csv("Data/" + i + "-USD-test.csv", names=["time", "open", "close", "high", "low", "volume"],index_col = "time")
    cryptos[i] = cryptos[i][x_names] #only keep relevant columns
    cryptos[i].columns = i + " " + cryptos[i].columns.values #add crypto name to column headers for later when the datasets are combined
    cryptos[i].index = pd.to_datetime(cryptos[i].index, unit="ms")

#the below lines calculate the return until the target date determined by "foresight" in the above parameters, and converts it to a 1 or 0 depending on whether
#or not it is above the buy threshold in the above parameters. Ex. if buy threshold is 0.05 then "target" will be one for a return of 5% or more
cryptos[predict]["target"] = cryptos[predict][predict + " " + y_name].shift(periods=-foresight) / cryptos[predict][predict + " " + y_name]
cryptos[predict]["target"] = (cryptos[predict]["target"] > 1 + buy_threshold).astype(int) #convert to
cryptos[predict].dropna(inplace=True)
print(cryptos[predict]["target"])

altcrypto = pd.DataFrame(index = cryptos[predict].index)
#this loop scales the data and puts the data of the non-target currencies together in a frame, for the TsDataProcessor function
for i in cryptos:
    cryptos[i] = TsDataProcessor.Scaler(cryptos[i], "target")
    if i != predict:
        altcrypto = pd.merge(cryptos[i],altcrypto, how='inner', left_index=True, right_index=True)


#Returns all the input frames together in one frame as "combined", and an array of training data. see TsDataProcessor for more
combined, sequential = TsDataProcessor.TsDataProcessor(cryptos[predict], altcrypto, target = "target", t=hindsight)

#separates the data into validation and training sets, Valpercent is the proportion of the total data used for validation

random.shuffle(sequential)



#the data is balanced so we have an equal number of buys and sells, otherwise the algorithm
# can get promising results just by never buying (always predicting 0), or always buying (always predicting 1) depending on the data
tester = TsDataProcessor.Balance(sequential)

tester_x, tester_y = TsDataProcessor.split(sequential)

Model.evaluate(tester_x,tester_y)





