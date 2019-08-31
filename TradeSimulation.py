import matplotlib.pyplot as plt

def simulate(p, TestPredictions, test):
    Prediction = []
    for sample in TestPredictions:
        for i in range(0,p.LabelCount):
            if sample[i] == max(sample):
                Prediction.append(i)

    Investment = 1000
    count = 0
    profitcount = 0
    portfolio = []

    for i in range(0,len(Prediction)):
        portfolio.append(Investment)
        if Prediction[i] != p.LabelCount-1:
            Investment += ((test[p.TargetTickers[Prediction[i]] + ' TargetPrice'].iloc[i]-test[p.TargetTickers[Prediction[i]] + ' Price'].iloc[i])/test[p.TargetTickers[Prediction[i]] + ' Price'].iloc[i])*10
            count+=1
            if ((test[p.TargetTickers[Prediction[i]] + ' TargetPrice'].iloc[i]-test[p.TargetTickers[Prediction[i]] + ' Price'].iloc[i])/test[p.TargetTickers[Prediction[i]] + ' Price'].iloc[i]) > 0:
                profitcount+= 1
        # elif Prediction[i] == 1:
        #     Investment += ((test[p.TargetTickers[0] + ' Price'].iloc[i] - test[p.TargetTickers[0] + ' TargetPrice'].iloc[i]) /
        #                    test[p.TargetTickers[0] + ' Price'].iloc[i]) * 10
        if Investment < 0:
            print('Bust after ' + str(i) + ' minutes.')
            break

    if Investment > 0:
        print('You made ' + str(int(Investment-1000)) + ' euros. ' + str(int((Investment-1000)/10)) + '% Return in ' + str(len(Prediction)) + ' days.')
        print(str(count) + ' trades with a ' + str(int((profitcount/count)*100)) + '% success rate. Max drawdown of ' +  str(int(((1000-min(portfolio))*100/1000))) + '%.')

    plt.plot(portfolio)
    plt.ylabel('Profit')
    plt.xlabel('Days')


