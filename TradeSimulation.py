import matplotlib.pyplot as plt

def simulate(p, TestPredictions, test):
    Prediction = []
    for sample in TestPredictions:
        for i in range(0,p['LabelCount']):
            if sample[i] == max(sample):
                Prediction.append(i)

    InitialInvestment = 1000
    Investment = InitialInvestment
    count = 0
    profitcount = 0
    portfolio = []

    for i in range(0,len(Prediction)):
        portfolio.append(Investment)
        if Prediction[i] != p['LabelCount']-1:
            ticker = p['TargetTickers'][Prediction[i]]
            FuturePrice = test[ticker + ' TargetPrice'].iloc[i]
            Price = test[ticker + ' Price'].iloc[i]
            TradeReturn = ((FuturePrice-Price)/Price)*10
            Investment += TradeReturn
            count+=1
            if TradeReturn > 0:
                profitcount+= 1
        # elif Prediction[i] == 1:
        #     Investment += ((test[p['TargetTickers'][0] + ' Price'].iloc[i] - test[p['TargetTickers'][0] + ' TargetPrice'].iloc[i]) /
        #                    test[p['TargetTickers'][0] + ' Price'].iloc[i]) * 10
        if Investment < 0:
            print('Bust after ' + str(i) + ' minutes.')
            break

    TotalProfit = Investment-InitialInvestment
    TotalReturn = (TotalProfit)/(InitialInvestment/100)
    MaxDrawdown = ((InitialInvestment-min(portfolio))*100/InitialInvestment)

    if Investment > 0:
        print('You made ' + str(int(TotalProfit)) + ' euros. ' + str(int(TotalReturn)) + '% Return in ' + str(len(Prediction)) + ' days.')
        print(str(count) + ' trades with a ' + str(int((profitcount/count)*100)) + '% success rate. Max drawdown of ' +  str(int(MaxDrawdown)) + '%.')

    plt.plot(portfolio)
    plt.ylabel('Profit')
    plt.xlabel('Days')


