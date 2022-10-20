import os
import pandas as pd

os.chdir('/Users/justusmulli/Projects/OmniTrade/Data')
exchange = 'GDAX.'
ticker = 'BTC'
all_data = pd.DataFrame()
for file in [f for f in os.listdir() if exchange in f]:
    data = pd.read_csv(file, dtype= {'<TIME>': str, '<DATE>': str})
    data.columns = list(map(lambda x: x.replace('<', '').replace('>', ''), data.columns))
    data.TICKER = list(map(lambda x: x.replace(exchange, ''), data.TICKER))
    all_data = all_data.append(data)

all_data['timestamp'] = list(map(lambda x : str(x)[:4]+'-'+str(x)[4:6]+'-'+str(x)[6:],  all_data.DATE))
all_data.timestamp = all_data.timestamp + ' '
all_data.timestamp = all_data.timestamp + \
                     list(map(lambda x: str(x)[:2] +
                                        ':' + str(x)[2:4] +
                                        ':' + str(x)[4:], all_data.TIME))
all_data.columns = list(map(lambda x: x.lower(), all_data.columns))
all_data.drop(columns=['date', 'time', 'per', 'ticker'], inplace=True)
all_data.rename(columns={'vol': 'volume'}, inplace = True)
all_data = all_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
all_data = all_data.sort_values('timestamp')
all_data.drop_duplicates().to_csv(f'/Users/justusmulli/Projects/OmniTrade/Data/Minute/{ticker}.csv', index=None)
