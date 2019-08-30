Prediction = []

for sample in TestPredictions:
    for i in range(0,p.LabelCount):
        if sample[i] == max(sample):
            Prediction.append(i)

Investment = 1000
count = 0
for i in range(len(Prediction)-1,-1,-1):
    if Prediction[i] == 0:
        Investment += ((test[p.TargetTickers[0] + ' TargetPrice'].iloc[i]-test[p.TargetTickers[0] + ' Price'].iloc[i])/test[p.TargetTickers[0] + ' Price'].iloc[i])*10
        # if (test[TargetTickers[0] + ' TargetPrice'].iloc[i]-test[TargetTickers[0] + ' Price'].iloc[i]) > 0:
        #     count+= 1
    elif Prediction[i] == 1:
        Investment += ((test[p.TargetTickers[0] + ' Price'].iloc[i] - test[p.TargetTickers[0] + ' TargetPrice'].iloc[i]) /
                       test[p.TargetTickers[0] + ' Price'].iloc[i]) * 10
    if Investment < 0:
        print('Bust after ' + str(i) + ' minutes.')
        break

if Investment > 0:
    print('You made ' + str(int(Investment-1000)) + ' euros. ' + str(int((Investment-1000)/10)) + '% Return in ' + str(int(len(Prediction)/6/24)) + ' days.')
    print(str(sum(Prediction)) + ' trades with a ' + str(int(count*100/sum(Prediction))) + '% success rate.')